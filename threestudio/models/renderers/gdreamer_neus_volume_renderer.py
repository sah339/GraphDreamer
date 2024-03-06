from dataclasses import dataclass
from functools import partial

import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.ops import chunk_batch, validate_empty_rays
from threestudio.utils.typing import *
from threestudio.utils.misc import cleanup

from nerfacc.volrend import render_visibility_from_alpha


def volsdf_density(sdf, inv_std):
    inv_std = inv_std.clamp(0.0, 80.0)
    beta = 1 / inv_std
    alpha = inv_std
    return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))


class LearnedVariance(nn.Module):
    def __init__(self, init_val):
        super(LearnedVariance, self).__init__()
        self.register_parameter("_inv_std", nn.Parameter(torch.tensor(init_val)))

    @property
    def inv_std(self):
        val = torch.exp(self._inv_std * 10.0)
        return val

    def forward(self, x):
        '''
            x: sdf
        '''
        return torch.ones_like(x) * self.inv_std.clamp(1.0e-6, 1.0e6)


@threestudio.register("gdreamer-neus-renderer")
class GDreamerNeuSRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        learned_variance_init: float = 0.3
        cos_anneal_end_steps: int = 20000
        use_volsdf: bool = False
        near_plane: float = 0.1
        far_plane: float = 4.0

        hard_split: bool = False
        estimator: str = "occgrid"
        grid_prune: bool = True
        prune_alpha_threshold: bool = True
        num_samples_per_ray_importance: int = 64  # importance sampling
        
    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.variance = LearnedVariance(self.cfg.learned_variance_init)
                
        if self.cfg.estimator == "occgrid":
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=32, levels=1
            )
            if not self.cfg.grid_prune:
                self.estimator.occs.fill_(True)
                self.estimator.binaries.fill_(True)
            self.render_step_size = (
                1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
            )            
            self.randomized = self.cfg.randomized
        elif self.cfg.estimator == "importance":
            self.estimator = ImportanceEstimator()
        else:
            raise NotImplementedError(
                "unknown estimator, should be in ['occgrid', 'importance']"
            )
        self.cos_anneal_ratio = 1.0
    
    def argmin_in_forward_softmin_in_backward(
            self,
            x: Float[Tensor, "Nr Di"]
        ) -> Float[Tensor, "*N Di"]:
        fw = F.softmin(1e16 * x, dim=-1)
        fw = torch.nan_to_num(fw, nan=1.0, posinf=1.0, neginf=0.0)
        bw = F.softmax(-x - (-x).max(dim=-1, keepdim=True).values, dim=-1)
        return fw.detach() + (bw - bw.detach())  # 0. and 1. with gradients
           
    def get_alpha(self, sdf, normal, dirs, dists):        
        alpha = torch.zeros_like(sdf)
        num_objects = sdf.shape[-1]
        
        for i in range(num_objects):
            inv_std = self.variance(sdf[..., i:i+1])
            
            if self.cfg.use_volsdf:
                alpha[..., i:i+1] = torch.abs(dists.detach()) * volsdf_density(sdf[..., i:i+1], inv_std)
            else:
                true_cos = (dirs * normal[..., i]).sum(-1, keepdim=True)
                # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
                # the cos value "not dead" at the beginning training iterations, for better convergence.
                iter_cos = -(
                    F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)
                    + F.relu(-true_cos) * self.cos_anneal_ratio
                )  # always non-positive

                # Estimate signed distances at section points
                estimated_next_sdf = sdf[..., i:i+1] + iter_cos * dists * 0.5
                estimated_prev_sdf = sdf[..., i:i+1] - iter_cos * dists * 0.5

                prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)

                p = prev_cdf - next_cdf
                c = prev_cdf

                alpha[..., i:i+1] = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha
    
    def forward_edge(
        self,
        rays_o: Float[Tensor, "B H W 3"],  # update this one
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        curr_obj_idx: int = -1,
        global_step: int = -1,
        curr_edge: List[int, ] = [-1, ],
        near_plane: Optional[Any] = -1., 
        far_plane: Optional[Any] = -1., 
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]
        num_objects = self.geometry.cfg.num_objects
        
        # (0) Pre-Sampling: 
        near_plane = self.cfg.near_plane if near_plane < 0 else near_plane
        far_plane = self.cfg.far_plane if far_plane < 0 else far_plane
        if self.cfg.estimator == "occgrid":
            
            def alpha_fn(t_starts, t_ends, ray_indices):
                t_starts, t_ends = t_starts[..., None], t_ends[..., None]
                t_origins = rays_o_flatten[ray_indices]
                t_positions = (t_starts + t_ends) / 2.0
                t_dirs = rays_d_flatten[ray_indices]
                positions = t_origins + t_dirs * t_positions
                if self.training:
                    sdf = self.geometry.forward_sdf(positions)
                else:
                    with torch.no_grad():
                        sdf = chunk_batch(
                            self.geometry.forward_sdf,
                            self.cfg.eval_chunk_size,
                            positions, 
                        )
                soft_Label = self.argmin_in_forward_softmin_in_backward(sdf)
                
                alpha = torch.zeros_like(sdf)  # [Nr, K]
                for i in range(num_objects):
                    inv_std = self.variance(sdf[..., i])
                    
                    if self.cfg.use_volsdf:
                        alpha[..., i] = self.render_step_size * volsdf_density(sdf[..., i], inv_std)
                    else:
                        estimated_next_sdf = sdf[..., i] - self.render_step_size * 0.5
                        estimated_prev_sdf = sdf[..., i] + self.render_step_size * 0.5
                        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                        next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                        p = prev_cdf - next_cdf
                        c = prev_cdf
                        alpha[..., i] = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                
                return (soft_Label * alpha)[..., curr_edge].sum(dim=-1)

            if not self.cfg.grid_prune:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        alpha_fn=None,
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0.0,
                    )
            else:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        alpha_fn=alpha_fn if self.cfg.prune_alpha_threshold else None,
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0.0,  # special design for object rendering
                    )
        elif self.cfg.estimator == "importance":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                if self.cfg.use_volsdf:
                    t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                    t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                    positions: Float[Tensor, "Nr Ns 3"] = (
                        t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                    )
                    with torch.no_grad():
                        geo_out = chunk_batch(
                            proposal_network,
                            self.cfg.eval_chunk_size,
                            positions.reshape(-1, 3),
                            output_normal=False,
                        )
                        inv_std = self.variance(geo_out["sdf"])
                        density = volsdf_density(geo_out["sdf"], inv_std)
                    return density.reshape(positions.shape[:2])
                else:
                    raise ValueError(
                        "Currently only VolSDF supports importance sampling."
                    )

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.geometry)],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=near_plane,
                far_plane=far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError
        ray_indices, t_starts_, t_ends_ = validate_empty_rays(ray_indices, t_starts_, t_ends_)
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions  # unscaled
        t_intervals = t_ends - t_starts

        """Calculation of geometry and color: """
        if self.training:
            geo_out = self.geometry(positions, output_normal=True, curr_edge=curr_edge, no_global=True)
            comp_rgb_bg = self.background(norm_func=self.geometry.normalize_points, dirs=rays_d, origins=rays_o, radius=self.cfg.radius)
            # comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=True,
                curr_edge=curr_edge, no_global=True
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, 
                norm_func=self.geometry.normalize_points, dirs=rays_d, origins=rays_o, radius=self.cfg.radius
            )
            # comp_rgb_bg = chunk_batch(
            #     self.background, self.cfg.eval_chunk_size, dirs=rays_d
            # )
        
        if bg_color is None:
            bg_color = comp_rgb_bg
        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)
        
        # grad or normal?
        sdf_OBJ = geo_out["sdf_OBJ"]
        normal_OBJ = geo_out["normal_OBJ"]
        alpha_OBJ = self.get_alpha(sdf_OBJ, normal_OBJ, t_dirs, t_intervals)
        soft_Labels = geo_out["soft_Label"]

        normal_G = geo_out["normal_edge"]
        sdf_grad_G = geo_out["sdf_grad_edge"]
        
        alpha_OBJ = soft_Labels * alpha_OBJ
        alpha_G = alpha_OBJ[..., curr_edge].sum(dim=-1, keepdim=True)
        sdf_G = (soft_Labels * sdf_OBJ).sum(-1)
        
        Labels = soft_Labels.detach()
        Labels = Labels.argmax(dim=-1)

        '''Get shading rgb: '''
        # (1) global: 
        shading_normal_g = geo_out["shading_normal_edge"]          
        if self.training:
            rgb_fg_G: Float[Tensor, "B ... 3"] = self.material(
                features=geo_out["features_edge"],
                viewdirs=t_dirs,
                positions=positions,
                shading_normal=shading_normal_g,
                light_positions=t_light_positions,
                **kwargs
            )    
        else:
            rgb_fg_G = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                features=geo_out["features_edge"],
                viewdirs=t_dirs,
                positions=positions,
                shading_normal=shading_normal_g,
                light_positions=t_light_positions,
                **kwargs
            )  
        # (2) objects:
        if self.training:
            rgb_fg_OBJ: Float[Tensor, "B ... K 3"] = self.material.forward_objects(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                features=geo_out["features_OBJ"],
                shading_normal=geo_out["shading_normal_OBJ"],
                **kwargs
            )
        else:
            rgb_fg_OBJ = chunk_batch(
                self.material.forward_objects,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                features=geo_out["features_OBJ"],
                shading_normal=geo_out["shading_normal_OBJ"],
                **kwargs
            )

        out = {
            "sdf_OBJ": sdf_OBJ,
            "sdf_G": sdf_G,
            "soft_Labels": soft_Labels,
        }

        # (1) Global rendering:
        with torch.no_grad():
            mask_g = render_visibility_from_alpha(
                alphas=alpha_G[..., 0],
                ray_indices=ray_indices,
                alpha_thre=0.001 if self.cfg.prune_alpha_threshold else 0.0,
                early_stop_eps=0.0001,
            )
        
        ray_indices_g, t_starts_g_, t_ends_g_ = ray_indices[mask_g], t_starts_[mask_g], t_ends_[mask_g]  # validate_empty_rays(
        t_starts_g, t_ends_g = t_starts_g_[..., None], t_ends_g_[..., None]
        t_origins_g = rays_o_flatten[ray_indices_g]
        t_dirs_g = rays_d_flatten[ray_indices_g]
        # t_light_positions = light_positions_flatten[ray_indices_g]
        t_positions_g = (t_starts_g + t_ends_g) / 2.0
        positions_g = t_origins_g + t_dirs_g * t_positions_g
        t_intervals_g = t_ends_g - t_starts_g
        
        alpha_g = alpha_G[mask_g]
        normal_g = normal_G[mask_g]  # not used in loss
        sdf_grad_g = sdf_grad_G[mask_g]  # not used in loss
        sdf_obj = sdf_OBJ[mask_g]
        
        ''''''
        weights_g: Float[Tensor, "Nr 1"]
        weights_g_, _ = nerfacc.render_weight_from_alpha(
            alpha_g[..., 0],
            ray_indices=ray_indices_g,
            n_rays=n_rays,
        )
        weights_g = weights_g_[..., None]
        opacity_g: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights_g[..., 0], values=None, ray_indices=ray_indices_g, n_rays=n_rays
        )
        # Global mask rendering:
        comp_mask_fg: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights_g[..., 0], values=soft_Labels.max(dim=-1, keepdim=True).values[mask_g], ray_indices=ray_indices_g, n_rays=n_rays
        )
        depth_g: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights_g[..., 0], values=t_positions_g, ray_indices=ray_indices_g, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights_g[..., 0], values=rgb_fg_G[mask_g], ray_indices=ray_indices_g, n_rays=n_rays
        )
        comp_rgb = comp_rgb_fg + bg_color * (1.0 - comp_mask_fg)
        # if self.training:
        #     comp_rgb = comp_rgb_fg + bg_color * (1.0 - comp_mask_fg)
        # else:
        #     comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity_g)
        
        out.update(
            {
                "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
                "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
                "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
                "opacity": opacity_g.view(batch_size, height, width, 1),
                "depth": depth_g.view(batch_size, height, width, 1),
                "comp_mask_fg": comp_mask_fg.view(batch_size, height, width, 1), #
            }
        )
        if self.training:
            out.update(
                {
                    "weights": weights_g,
                    "t_points": t_positions_g,
                    "t_intervals": t_intervals_g,
                    "t_dirs": t_dirs_g,
                    "ray_indices": ray_indices_g,
                    "points": positions_g,
                    # **geo_out, 
                    "sdf": sdf_obj,  # For constraint at most one element in SDF vector is < 0.
                    # "alpha": alpha_g,
                    "normal": normal_g,
                    "sdf_grad": sdf_grad_g, 
                }
            )
        if "normal_edge" in geo_out:
            comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                weights_g[..., 0],
                values=normal_g,
                ray_indices=ray_indices_g,
                n_rays=n_rays,
            )
            comp_normal = F.normalize(comp_normal, dim=-1)  # [-1, 1]
            if self.training:  # for omni normal loss
                out.update(
                    {"comp_normal": comp_normal.view(batch_size, height, width, 3),}
                )
            else:   # for visualization: [0, 1]
                comp_normal = (comp_normal + 1.0) / 2.0  * opacity_g
                comp_normal += bg_color * (1.0 - opacity_g)
                out.update(
                    {"comp_normal": comp_normal.view(batch_size, height, width, 3),}
                )
        out.update({"inv_std": self.variance.inv_std})
        
        # (2) Object rendering: 
        comp_rgb_fg_obj = []
        comp_rgb_obj = []
        comp_mask_obj = []
        comp_normal_obj = []
        opacity_obj = []
        
        weights_obj = []
        depth_obj = []
        geo_out_obj = []
        t_dirs_obj = []
        points_obj = []

        for i in range(num_objects):
            
            if self.training and curr_obj_idx in range(num_objects + 1) and i != curr_obj_idx:
                comp_rgb_obj.append(None)
                comp_rgb_fg_obj.append(None)
                comp_mask_obj.append(None)
                if "normal_OBJ" in geo_out:
                    comp_normal_obj.append(None)
                opacity_obj.append(None)
                depth_obj.append(None)
                weights_obj.append(None)
                geo_out_obj.append(None)
                t_dirs_obj.append(None)
                points_obj.append(None)

            else:
                
                mask_i = Labels == i  # [Nr, ]
                if not self.cfg.hard_split:
                    mask_i = mask_i == mask_i
                
                ray_indices_i, t_starts_i_, t_ends_i_ = ray_indices[mask_i], t_starts_[mask_i], t_ends_[mask_i]
                t_starts_i, t_ends_i = t_starts_i_[..., None], t_ends_i_[..., None]
                t_positions_i = (t_starts_i + t_ends_i) / 2.0
                t_dirs_i = rays_d_flatten[ray_indices_i]
                t_origins_i = rays_o_flatten[ray_indices_i]
                positions_i = t_origins_i + t_dirs_i * t_positions_i
                
                sdf_i = sdf_OBJ[mask_i, i:i+1]
                alpha_i = alpha_OBJ[mask_i, i:i+1]
                sdf_grad_i = geo_out["sdf_grad_OBJ"][mask_i, ...][..., i:i+1]
                normal_i = normal_OBJ[mask_i, ...][..., i]
                soft_label_i = soft_Labels[mask_i, i:i+1]
                sdf_OBJ_i = sdf_OBJ[mask_i]
                rgb_fg_i = rgb_fg_OBJ[mask_i, ...][..., i, :]
                
                '''seletion:'''
                if mask_i.sum().item() > 0: 
                    with torch.no_grad():
                        m_i = render_visibility_from_alpha(
                            alphas=alpha_i[..., 0],
                            ray_indices=ray_indices[mask_i],
                            alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                            early_stop_eps=0.0001,
                        )
                        if m_i.sum().item() == 0:
                            threestudio.info(f"Samples classified to object {i} are all invisible!")
                    
                    ray_indices_i, t_starts_i_, t_ends_i_ = ray_indices_i[m_i], t_starts_i_[m_i], t_ends_i_[m_i]
                    t_starts_i, t_ends_i = t_starts_i_[..., None], t_ends_i_[..., None]
                    t_positions_i = (t_starts_i + t_ends_i) / 2.0
                    t_dirs_i = rays_d_flatten[ray_indices_i]
                    t_origins_i = rays_o_flatten[ray_indices_i]
                    positions_i = t_origins_i + t_dirs_i * t_positions_i
                    
                    sdf_i = sdf_i[m_i]
                    alpha_i = alpha_i[m_i]
                    rgb_fg_i = rgb_fg_i[m_i]
                    sdf_grad_i = sdf_grad_i[m_i]
                    normal_i = normal_i[m_i]
                    soft_label_i = soft_label_i[m_i]
                    sdf_OBJ_i = sdf_OBJ_i[m_i]
                else:
                    threestudio.info(f"No samples are classified to object {i}!")
                ''''''

                weights_i: Float[Tensor, "Nr 1"]
                weights_i_, _ = nerfacc.render_weight_from_alpha(
                        alpha_i[..., 0],
                        ray_indices=ray_indices_i,
                        n_rays=n_rays,)
                weights_i = weights_i_[..., None]

                opacity_i: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                    weights_i[..., 0], values=None, ray_indices=ray_indices_i, n_rays=n_rays
                )
                opacity_obj.append(opacity_i.view(batch_size, height, width, 1))

                depth_i: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                    weights_i[..., 0], values=t_positions_i, ray_indices=ray_indices_i, n_rays=n_rays)
                depth_obj.append(depth_i.view(batch_size, height, width, 1))

                comp_mask_i: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                    weights_i[..., 0], values=soft_label_i, ray_indices=ray_indices_i, n_rays=n_rays
                )
                
                comp_mask_obj.append(comp_mask_i.view(batch_size, height, width, 1))

                '''rendering for SDS guidance:'''
                comp_rgb_fg_i: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
                        weights_i[..., 0], values=rgb_fg_i, ray_indices=ray_indices_i, n_rays=n_rays)
                comp_rgb_fg_obj.append(comp_rgb_fg_i.view(batch_size, height, width, -1))
                
                comp_rgb_i = comp_rgb_fg_i + bg_color * (1.0 - comp_mask_i)
                comp_rgb_obj.append(comp_rgb_i.view(batch_size, height, width, -1))

                if self.training:
                    weights_obj.append(weights_i)
                    geo_out_i = {
                        "sdf_obj_i": sdf_OBJ_i,
                        "normal_i": normal_i, 
                        "sdf_grad_i": sdf_grad_i,
                    }
                    geo_out_obj.append(geo_out_i)
                    t_dirs_obj.append(t_dirs_i)
                    points_obj.append(positions_i)
                if "normal_OBJ" in geo_out:
                    comp_normal_i: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights_i[..., 0], values=normal_i, ray_indices=ray_indices_i, n_rays=n_rays,
                    )
                    comp_normal_i = F.normalize(comp_normal_i, dim=-1)  # -1, 1
                    
                    if self.training:  # for omni normal loss
                        comp_normal_obj.append(comp_normal_i.view(batch_size, height, width, 3))
                    else:
                        comp_normal_i = (comp_normal_i + 1.0) / 2.0 * opacity_i 
                        comp_normal_i += bg_color * (1.0 - opacity_i)
                        comp_normal_obj.append(comp_normal_i.view(batch_size, height, width, 3))
                
        out.update(
            {
                "comp_rgb_obj": comp_rgb_obj,
                "comp_rgb_fg_obj": comp_rgb_fg_obj,
                "opacity_obj": opacity_obj, 
                "depth_obj": depth_obj,
                "comp_mask_obj": comp_mask_obj,
            }
        )
        if self.training:
            out.update(
            {
                "weights_obj": weights_obj, 
                "geo_out_obj": geo_out_obj,
                "t_dirs_obj": t_dirs_obj,
                "points_obj": points_obj,
            }
        )
        if "normal_OBJ" in geo_out:
            out.update(
                {
                    "comp_normal_obj": comp_normal_obj,
                }
            )
        
        return out
            
    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],  # update this one
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        curr_obj_idx: int = -1,
        global_step: int = -1,
        no_global: bool = False,
        no_local: bool = False,
        near_plane: Optional[Any] = -1., 
        far_plane: Optional[Any] = -1., 
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]
        num_objects = self.geometry.cfg.num_objects
        
        # (0) Pre-Sampling: ==================================================================================================================================
        near_plane = self.cfg.near_plane if near_plane < 0 else near_plane
        far_plane = self.cfg.far_plane if far_plane < 0 else far_plane

        if self.cfg.estimator == "occgrid":
            
            def alpha_fn(t_starts, t_ends, ray_indices):
                t_starts, t_ends = t_starts[..., None], t_ends[..., None]
                t_origins = rays_o_flatten[ray_indices]
                t_positions = (t_starts + t_ends) / 2.0
                t_dirs = rays_d_flatten[ray_indices]
                positions = t_origins + t_dirs * t_positions
                if self.training:
                    sdf = self.geometry.forward_sdf(positions)
                else:
                    with torch.no_grad():
                        sdf = chunk_batch(
                            self.geometry.forward_sdf,
                            self.cfg.eval_chunk_size,
                            positions, 
                        )
                soft_Label = self.argmin_in_forward_softmin_in_backward(sdf)
                
                alpha = torch.zeros_like(sdf)  # [Nr, K]
                for i in range(num_objects):
                    inv_std = self.variance(sdf[..., i])
                    
                    if self.cfg.use_volsdf:
                        alpha[..., i] = self.render_step_size * volsdf_density(sdf[..., i], inv_std)
                    else:
                        estimated_next_sdf = sdf[..., i] - self.render_step_size * 0.5
                        estimated_prev_sdf = sdf[..., i] + self.render_step_size * 0.5
                        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                        next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                        p = prev_cdf - next_cdf
                        c = prev_cdf
                        alpha[..., i] = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                
                return (soft_Label * alpha).sum(dim=-1)

            if not self.cfg.grid_prune:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        alpha_fn=None,
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0.0,
                    )
            else:
                with torch.no_grad():
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        alpha_fn=alpha_fn if self.cfg.prune_alpha_threshold else None,
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_step_size=self.render_step_size,
                        alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                        early_stop_eps=0.0,  # special design for object rendering
                    )

        elif self.cfg.estimator == "importance":

            def prop_sigma_fn(
                t_starts: Float[Tensor, "Nr Ns"],
                t_ends: Float[Tensor, "Nr Ns"],
                proposal_network,
            ):
                if self.cfg.use_volsdf:
                    t_origins: Float[Tensor, "Nr 1 3"] = rays_o_flatten.unsqueeze(-2)
                    t_dirs: Float[Tensor, "Nr 1 3"] = rays_d_flatten.unsqueeze(-2)
                    positions: Float[Tensor, "Nr Ns 3"] = (
                        t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
                    )
                    with torch.no_grad():
                        geo_out = chunk_batch(
                            proposal_network,
                            self.cfg.eval_chunk_size,
                            positions.reshape(-1, 3),
                            output_normal=False,
                        )
                        inv_std = self.variance(geo_out["sdf"])
                        density = volsdf_density(geo_out["sdf"], inv_std)
                    return density.reshape(positions.shape[:2])
                else:
                    raise ValueError(
                        "Currently only VolSDF supports importance sampling."
                    )

            t_starts_, t_ends_ = self.estimator.sampling(
                prop_sigma_fns=[partial(prop_sigma_fn, proposal_network=self.geometry)],
                prop_samples=[self.cfg.num_samples_per_ray_importance],
                num_samples=self.cfg.num_samples_per_ray,
                n_rays=n_rays,
                near_plane=near_plane,
                far_plane=far_plane,
                sampling_type="uniform",
                stratified=self.randomized,
            )
            ray_indices = (
                torch.arange(n_rays, device=rays_o_flatten.device)
                .unsqueeze(-1)
                .expand(-1, t_starts_.shape[1])
            )
            ray_indices = ray_indices.flatten()
            t_starts_ = t_starts_.flatten()
            t_ends_ = t_ends_.flatten()
        else:
            raise NotImplementedError

        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions  # unscaled
        t_intervals = t_ends - t_starts

        """# All training-used calculation of geometry and color is done here: """
        if self.training:
            geo_out = self.geometry(positions, output_normal=True)
            comp_rgb_bg = self.background(norm_func=self.geometry.normalize_points, dirs=rays_d, origins=rays_o, radius=self.cfg.radius)
            # comp_rgb_bg = self.background(dirs=rays_d)
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=True,
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, 
                norm_func=self.geometry.normalize_points, dirs=rays_d, origins=rays_o, radius=self.cfg.radius
            )
            # comp_rgb_bg = chunk_batch(
            #     self.background, self.cfg.eval_chunk_size, dirs=rays_d
            # )
        
        if bg_color is None:
            bg_color = comp_rgb_bg
        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)
        
        # grad or normal?
        sdf_OBJ = geo_out["sdf_OBJ"]
        normal_OBJ = geo_out["normal_OBJ"]
        alpha_OBJ = self.get_alpha(sdf_OBJ, normal_OBJ, t_dirs, t_intervals)
        soft_Labels = geo_out["soft_Label"]

        normal_G = geo_out["normal_g"]
        sdf_grad_G = geo_out["sdf_grad_g"]
        
        alpha_OBJ = soft_Labels * alpha_OBJ
        alpha_G = alpha_OBJ.sum(dim=-1, keepdim=True)
        sdf_G = (soft_Labels * sdf_OBJ).sum(-1)
        
        Labels = soft_Labels.detach()
        Labels = Labels.argmax(dim=-1)

        '''Get shading rgb: '''
        # (1) global: 
        shading_normal_g = geo_out["shading_normal_g"]
        if self.training:
            rgb_fg_G: Float[Tensor, "B ... 3"] = self.material(
                features=geo_out["features_g"],
                viewdirs=t_dirs,
                positions=positions,
                shading_normal=shading_normal_g,
                light_positions=t_light_positions,
                **kwargs
            )    
        else:
            rgb_fg_G = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                features=geo_out["features_g"],
                viewdirs=t_dirs,
                positions=positions,
                shading_normal=shading_normal_g,
                light_positions=t_light_positions,
                **kwargs
            )
        # (2) objects:        
        if self.training:
            rgb_fg_OBJ: Float[Tensor, "B ... K 3"] = self.material.forward_objects(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                features=geo_out["features_OBJ"],
                shading_normal=geo_out["shading_normal_OBJ"],
                **kwargs
            )    
        else:
            rgb_fg_OBJ = chunk_batch(
                self.material.forward_objects,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                features=geo_out["features_OBJ"],
                shading_normal=geo_out["shading_normal_OBJ"],
                **kwargs
            )

        out = {
            "sdf_OBJ": sdf_OBJ,
            "sdf_G": sdf_G,
            "soft_Labels": soft_Labels,
        }
        # (1) Global rendering:
        if not no_global:
            with torch.no_grad():
                mask_g = render_visibility_from_alpha(
                        # alphas=alphas,
                        alphas=alpha_G[..., 0],
                        ray_indices=ray_indices,
                        alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                        early_stop_eps=0.0001,
                    )
            
            ray_indices_g, t_starts_g_, t_ends_g_ = ray_indices[mask_g], t_starts_[mask_g], t_ends_[mask_g]  # validate_empty_rays(

            t_starts_g, t_ends_g = t_starts_g_[..., None], t_ends_g_[..., None]
            t_origins_g = rays_o_flatten[ray_indices_g]
            t_dirs_g = rays_d_flatten[ray_indices_g]
            # t_light_positions = light_positions_flatten[ray_indices_g]
            t_positions_g = (t_starts_g + t_ends_g) / 2.0
            positions_g = t_origins_g + t_dirs_g * t_positions_g
            t_intervals_g = t_ends_g - t_starts_g
            
            sdf_obj = sdf_OBJ[mask_g]
            alpha_g = alpha_G[mask_g]
            normal_g = normal_G[mask_g]  # not used in loss
            sdf_grad_g = sdf_grad_G[mask_g]  # not used in loss
            
            ''''''
            weights_g: Float[Tensor, "Nr 1"]
            weights_g_, _ = nerfacc.render_weight_from_alpha(
                alpha_g[..., 0],
                ray_indices=ray_indices_g,
                n_rays=n_rays,
            )
            weights_g = weights_g_[..., None]
            opacity_g: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                weights_g[..., 0], values=None, ray_indices=ray_indices_g, n_rays=n_rays
            )
            # Global mask rendering:
            comp_mask_fg: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                weights_g[..., 0], values=soft_Labels.max(dim=-1, keepdim=True).values[mask_g], ray_indices=ray_indices_g, n_rays=n_rays
            )
            depth_g: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                weights_g[..., 0], values=t_positions_g, ray_indices=ray_indices_g, n_rays=n_rays
            )
            comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
                weights_g[..., 0], values=rgb_fg_G[mask_g], ray_indices=ray_indices_g, n_rays=n_rays
            )
            
            comp_rgb = comp_rgb_fg + bg_color * (1.0 - comp_mask_fg)

            out.update(
                {
                    "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
                    "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
                    "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
                    "opacity": opacity_g.view(batch_size, height, width, 1),
                    "depth": depth_g.view(batch_size, height, width, 1),
                    "comp_mask_fg": comp_mask_fg.view(batch_size, height, width, 1), #
                }
            )
            
            if self.training:
                out.update(
                    {
                        "weights": weights_g,
                        "t_points": t_positions_g,
                        "t_intervals": t_intervals_g,
                        "t_dirs": t_dirs_g,
                        "ray_indices": ray_indices_g,
                        "points": positions_g,
                        # 
                        "sdf": sdf_obj,  # For constraint at most one element in SDF vector is < 0.
                        "normal": normal_g,
                        "sdf_grad": sdf_grad_g, 
                    }
                )
            if "normal_g" in geo_out:
                comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                    weights_g[..., 0],
                    values=normal_g,
                    ray_indices=ray_indices_g,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)  # [-1, 1]
                if self.training: 
                    out.update(
                        {"comp_normal": comp_normal.view(batch_size, height, width, 3),}
                    )
                else:   # for visualization: [0, 1]
                    comp_normal = (comp_normal + 1.0) / 2.0  * opacity_g
                    comp_normal += bg_color * (1.0 - opacity_g)
                    out.update(
                        {"comp_normal": comp_normal.view(batch_size, height, width, 3),}
                    )
            out.update({"inv_std": self.variance.inv_std})
            
            if no_local:
                return out
                        
        # (2) Object rendering: 
        comp_rgb_fg_obj = []
        comp_rgb_obj = []
        comp_mask_obj = []
        comp_normal_obj = []
        opacity_obj = []
        
        weights_obj = []
        depth_obj = []
        geo_out_obj = []
        t_dirs_obj = []
        points_obj = []

        for i in range(num_objects):
            if self.training and curr_obj_idx in range(num_objects + 1) and i != curr_obj_idx:

                comp_rgb_obj.append(None)
                comp_rgb_fg_obj.append(None)
                comp_mask_obj.append(None)
                if "normal_OBJ" in geo_out:
                    comp_normal_obj.append(None)
                opacity_obj.append(None)
                depth_obj.append(None)
                weights_obj.append(None)
                geo_out_obj.append(None)
                t_dirs_obj.append(None)
                points_obj.append(None)

            else:
                
                mask_i = Labels == i
                if not self.cfg.hard_split:
                    mask_i = mask_i == mask_i
                
                ray_indices_i, t_starts_i_, t_ends_i_ = ray_indices[mask_i], t_starts_[mask_i], t_ends_[mask_i]
                t_starts_i, t_ends_i = t_starts_i_[..., None], t_ends_i_[..., None]
                t_positions_i = (t_starts_i + t_ends_i) / 2.0
                t_dirs_i = rays_d_flatten[ray_indices_i]
                
                t_origins_i = rays_o_flatten[ray_indices_i]
                positions_i = t_origins_i + t_dirs_i * t_positions_i
                
                sdf_i = sdf_OBJ[mask_i, i:i+1]
                alpha_i = alpha_OBJ[mask_i, i:i+1]
                rgb_fg_i = rgb_fg_OBJ[mask_i, ...][..., i, :]
                sdf_grad_i = geo_out["sdf_grad_OBJ"][mask_i, ...][..., i:i+1]
                normal_i = normal_OBJ[mask_i, ...][..., i]
                soft_label_i = soft_Labels[mask_i, i:i+1]
                
                sdf_OBJ_i = sdf_OBJ[mask_i]
                
                '''seletion:'''
                if mask_i.sum().item() > 0: 
                    with torch.no_grad():
                        m_i = render_visibility_from_alpha(
                            alphas=alpha_i[..., 0],
                            ray_indices=ray_indices[mask_i],
                            alpha_thre=0.01 if self.cfg.prune_alpha_threshold else 0.0,
                            early_stop_eps=0.0001,  # <---
                        )
                        if m_i.sum().item() == 0:
                            threestudio.info(f"Samples classified to object {i} are all invisible!")
                    
                    ray_indices_i, t_starts_i_, t_ends_i_ = ray_indices_i[m_i], t_starts_i_[m_i], t_ends_i_[m_i]
                    t_starts_i, t_ends_i = t_starts_i_[..., None], t_ends_i_[..., None]
                    t_positions_i = (t_starts_i + t_ends_i) / 2.0
                    t_dirs_i = rays_d_flatten[ray_indices_i]
                    
                    t_origins_i = rays_o_flatten[ray_indices_i]
                    positions_i = t_origins_i + t_dirs_i * t_positions_i
                    
                    sdf_i = sdf_i[m_i]
                    alpha_i = alpha_i[m_i]
                    rgb_fg_i = rgb_fg_i[m_i]
                    sdf_grad_i = sdf_grad_i[m_i]
                    normal_i = normal_i[m_i]
                    soft_label_i = soft_label_i[m_i]

                    sdf_OBJ_i = sdf_OBJ_i[m_i]
                
                else:
                    threestudio.info(f"No samples are classified to object {i}!")

                weights_i: Float[Tensor, "Nr 1"]
                weights_i_, _ = nerfacc.render_weight_from_alpha(
                        alpha_i[..., 0],
                        ray_indices=ray_indices_i,
                        n_rays=n_rays,)
                weights_i = weights_i_[..., None]

                opacity_i: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                    weights_i[..., 0], values=None, ray_indices=ray_indices_i, n_rays=n_rays
                )
                opacity_obj.append(opacity_i.view(batch_size, height, width, 1))

                depth_i: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                    weights_i[..., 0], values=t_positions_i, ray_indices=ray_indices_i, n_rays=n_rays)
                depth_obj.append(depth_i.view(batch_size, height, width, 1))

                comp_mask_i: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
                    weights_i[..., 0], values=soft_label_i, ray_indices=ray_indices_i, n_rays=n_rays
                )
                
                comp_mask_obj.append(comp_mask_i.view(batch_size, height, width, 1))

                '''rendering for SDS guidance:'''
                comp_rgb_fg_i: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
                        weights_i[..., 0], values=rgb_fg_i, ray_indices=ray_indices_i, n_rays=n_rays)
                comp_rgb_fg_obj.append(comp_rgb_fg_i.view(batch_size, height, width, -1))
                
                comp_rgb_i = comp_rgb_fg_i + bg_color * (1.0 - comp_mask_i)
                comp_rgb_obj.append(comp_rgb_i.view(batch_size, height, width, -1))

                if self.training:
                    weights_obj.append(weights_i)
                    geo_out_i = {
                        "sdf_obj_i": sdf_OBJ_i,
                        "normal_i": normal_i, 
                        "sdf_grad_i": sdf_grad_i,
                    }
                    geo_out_obj.append(geo_out_i)
                    t_dirs_obj.append(t_dirs_i)
                    points_obj.append(positions_i)
                
                if "normal_OBJ" in geo_out:
                    comp_normal_i: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights_i[..., 0], values=normal_i, ray_indices=ray_indices_i, n_rays=n_rays,
                    )
                    comp_normal_i = F.normalize(comp_normal_i, dim=-1)  # -1, 1
                    
                    if self.training: 
                        comp_normal_obj.append(comp_normal_i.view(batch_size, height, width, 3))
                    else:
                        comp_normal_i = (comp_normal_i + 1.0) / 2.0 * opacity_i 
                        comp_normal_i += bg_color * (1.0 - opacity_i)
                        comp_normal_obj.append(comp_normal_i.view(batch_size, height, width, 3))
                
        out.update(
            {
                "comp_rgb_obj": comp_rgb_obj,
                "comp_rgb_fg_obj": comp_rgb_fg_obj,
                "opacity_obj": opacity_obj, 
                "depth_obj": depth_obj,
                "comp_mask_obj": comp_mask_obj,
            }
        )
        if self.training:
            out.update(
            {
                "weights_obj": weights_obj, 
                "geo_out_obj": geo_out_obj,
                "t_dirs_obj": t_dirs_obj,
                "points_obj": points_obj,
            }
        )
        if "normal_OBJ" in geo_out:
            out.update(
                {
                    "comp_normal_obj": comp_normal_obj,
                }
            )
        
        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False,
    ) -> None:
        self.cos_anneal_ratio = (
            1.0
            if self.cfg.cos_anneal_end_steps == 0
            else min(1.0, global_step / self.cfg.cos_anneal_end_steps)
        )
        if self.cfg.estimator == "occgrid":
            if self.cfg.grid_prune:
                
                def occ_eval_fn(x):
                    num_objects = self.geometry.cfg.num_objects
                    sdf = self.geometry.forward_sdf(x)
                    soft_label = self.argmin_in_forward_softmin_in_backward(sdf)
                    alpha = torch.zeros_like(sdf)  # [Nr, K]
                    for i in range(num_objects):
                        inv_std = self.variance(sdf[..., i])
                        if self.cfg.use_volsdf:
                            alpha[..., i] = self.render_step_size * volsdf_density(sdf[..., i], inv_std)
                        else:
                            estimated_next_sdf = sdf[..., i] - self.render_step_size * 0.5
                            estimated_prev_sdf = sdf[..., i] + self.render_step_size * 0.5
                            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
                            next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)
                            p = prev_cdf - next_cdf
                            c = prev_cdf
                            alpha[..., i] = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
                    return (soft_label * alpha).sum(dim=-1)                    

                if self.training and not on_load_weights:
                    self.estimator.update_every_n_steps(
                        step=global_step, occ_eval_fn=occ_eval_fn
                    )

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()