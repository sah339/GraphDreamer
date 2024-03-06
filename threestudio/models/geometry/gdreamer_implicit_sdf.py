import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank, get_device
from threestudio.utils.typing import *
from threestudio.utils.ops import scale_tensor


@threestudio.register("gdreamer-implicit-sdf")
class GDreamerImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        num_objects: int = 2
        n_input_dims: int = 3
        n_feature_dims: int = 3

        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[str] = "finite_difference"
        finite_difference_normal_eps: Union[float, str] = 0.01
        
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None
        
        sdf_center_dispersion: Any = 0.2
        sdf_center_init_up: Any = 0.  # +z
        sdf_center_init_front: Any = 0.  # +x
        sdf_center_init_right: Any = 0.  # +y
        center_params: List[List[float]] = field(default_factory=lambda: [[],])
        radius_params: List[float] = field(default_factory=lambda: [])

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

        '''for mesh optimization:'''
        extract_mesh: bool = False
        isosurface_resolution: int = 128
        isosurface_deformable_grid: bool = True
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01
        fix_geometry: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        num_objects = self.cfg.num_objects
        self.num_objects = num_objects

        self.encoding = []
        for i in range(num_objects):
            setattr(self, f'encoding_{i}', 
                    get_encoding(self.cfg.n_input_dims, self.cfg.pos_encoding_config)
            )
            self.encoding.append(getattr(self, f'encoding_{i}'))
        
        n_enc_dims = self.encoding[0].n_output_dims
        
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                n_enc_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
        
        sdf_network_config = self.cfg.mlp_network_config
        self.sdf_network = get_mlp(
            n_enc_dims, 1, sdf_network_config
        )

        if num_objects == 1:
            self.cfg.sdf_bias = "sphere"
            self.cfg.sdf_bias_params = 0.5
            
        if self.cfg.sdf_bias == "learning":
            
            if len(self.cfg.center_params) == num_objects:
                center_lens = (len(c) for c in self.cfg.center_params)
                if all(c_len == 3 for c_len in center_lens):
                    center_params = self.cfg.center_params
                    center_params = torch.tensor(center_params, dtype=torch.float32).reshape(num_objects, 3)
            else:
                random.seed("GGG")
                center_params = torch.randn([num_objects, 3], dtype=torch.float32) #

                center_dispersion: List[float] = (
                    [self.cfg.sdf_center_dispersion, ] * num_objects if isinstance(self.cfg.sdf_center_dispersion, float) 
                    else self.cfg.sdf_center_dispersion
                )
                assert len(center_dispersion) == num_objects
                center_dispersion = torch.tensor(center_dispersion, dtype=torch.float32)
                center_params *= center_dispersion.view(num_objects, 1)
                
                center_params[:, -1] *= 0. # z-axis
                if self.cfg.sdf_center_init_up is not None:
                    upflag = False
                    if isinstance(self.cfg.sdf_center_init_up, float):
                        center_init_up: List[float] = [self.cfg.sdf_center_init_up, ] * num_objects
                        upflag = True
                    elif len(self.cfg.sdf_center_init_up) == num_objects:  
                        center_init_up = self.cfg.sdf_center_init_up
                        upflag = True
                    if upflag:
                        center_init_up = torch.tensor(center_init_up, dtype=torch.float32)
                        # print("center_init_up:", center_init_up)
                        center_params[:, -1] += center_init_up
                
                if self.cfg.sdf_center_init_front is not None:
                    fflag = False
                    if isinstance(self.cfg.sdf_center_init_front, float):
                        center_init_front: List[float] = [self.cfg.sdf_center_init_front, ] * num_objects
                        fflag = True
                    elif len(self.cfg.sdf_center_init_front) == num_objects:
                        center_init_front: List[float] = self.cfg.sdf_center_init_front
                        fflag = True
                    if fflag:
                        center_init_front = torch.tensor(center_init_front, dtype=torch.float32)
                        # print("center_init_front:", center_init_front)
                        center_params[:, 0] += center_init_front

                if self.cfg.sdf_center_init_right is not None:
                    rflag = False
                    if isinstance(self.cfg.sdf_center_init_right, float):
                        center_init_right: List[float] = [self.cfg.sdf_center_init_right, ] * num_objects
                        rflag = True
                    elif len(self.cfg.sdf_center_init_right) == num_objects:
                        center_init_right: List[float] = self.cfg.sdf_center_init_right
                        rflag = True
                    if rflag:
                        center_init_right = torch.tensor(center_init_right, dtype=torch.float32)
                        # print("center_init_right:", center_init_right)
                        center_params[:, 1] += center_init_right            
            
            if len(self.cfg.radius_params) == num_objects:
                radius_params = self.cfg.radius_params
            else:
                radius_params = [0.5, ] * num_objects
            
            self.sdf_center_params = center_params
            # [print(f"Initial center {i}: ", self.sdf_center_params[i]) for i in range(num_objects)]
            self.sdf_radius_params = radius_params
        
        if self.cfg.extract_mesh:
            if self.cfg.isosurface_deformable_grid:
                assert (
                    self.cfg.isosurface_method == "mt"
                ), "isosurface_deformable_grid only works with mt"
                self.deformation_network = get_mlp(
                    n_enc_dims, 3, self.cfg.mlp_network_config
                )
        self.finite_difference_normal_eps: Optional[float] = None

    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        if self.cfg.sdf_bias != 0.0:
            threestudio.warn(
                "shape_init and sdf_bias are both specified, which may lead to unexpected results."
            )

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            from pysdf import SDF

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm

        for _ in tqdm(
            range(1000),
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            points_rand = (
                torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
            )
            # sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            sdf_gt = sdf_pred.detach() * 0.
            for i in range(sdf_pred.shape[-1]):
                sdf_gt[..., i:i+1] = get_gt_sdf(points_rand)

            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)
    
    def isosurface(self) -> Mesh:  
        # return cached mesh if fix_geometry is True to save computation
        if self.cfg.fix_geometry and self.mesh is not None:
            return self.mesh
        
        mesh = self.isosurface_helper(self.sdf, self.deformation)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, self.isosurface_bbox
        )
        if self.cfg.isosurface_remove_outliers:
            mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)
        self.mesh = mesh
        return mesh
    
    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], 
        sdf: Float[Tensor, "*N K"], 
        curr_cls: int=-1,
    ) -> Float[Tensor, "*N K"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(dim=-1, keepdim=True).sqrt() - 1.0  
            # pseudo signed distance of an ellipsoid
        
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params  # 0.5
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        
        elif self.cfg.sdf_bias == "learning":
            if curr_cls == -1:
                radius = torch.max(self.sdf_radius_params)
                radius = max([radius, 0.5])
                sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
            else:
                center = self.sdf_center_params[curr_cls].to(points)
                radius = self.sdf_radius_params[curr_cls]
                
                if not isinstance(radius, float):  # Ellipsoid
                    assert len(radius) == 3
                    radius = torch.tensor(radius).to(points)
                    sdf_bias = (((points - center) / radius) ** 2).sum(dim=-1, keepdim=True).sqrt() - 1.0
                else:
                    sdf_bias = ((points - center)**2).sum(dim=-1, keepdim=True).sqrt() - radius

        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias
        
    def argmin_in_forward_softmin_in_backward(
            self,
            x: Float[Tensor, "Nr Di"]
        ) -> Float[Tensor, "*N Di"]:
        fw = F.softmin(1e16 * x, dim=-1)
        fw = torch.nan_to_num(fw, nan=1.0, posinf=1.0, neginf=0.0)
        # bw = F.softmin(x, dim=-1)
        bw = F.softmax(-x - (-x).max(dim=-1, keepdim=True).values, dim=-1)
        return fw.detach() + (bw - bw.detach())  # 0. and 1. with gradients
    
    def get_normal(self, points_unscaled, enc, sdf, idx, grad_enabled):  # single entry
        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            assert self.finite_difference_normal_eps is not None
            eps: float = self.finite_difference_normal_eps
           
            if self.cfg.normal_type == "finite_difference_laplacian":
                offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                    [
                        [eps, 0.0, 0.0],
                        [-eps, 0.0, 0.0],
                        [0.0, eps, 0.0],
                        [0.0, -eps, 0.0],
                        [0.0, 0.0, eps],
                        [0.0, 0.0, -eps],
                    ]
                ).to(points_unscaled)
                points_offset: Float[Tensor, "... 6 3"] = (
                    points_unscaled[..., None, :] + offsets
                ).clamp(-self.cfg.radius, self.cfg.radius)
                
                if isinstance(idx, int):
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf_i(points_offset, idx)
                elif isinstance(idx, list):
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf_edge(points_offset, idx)
                elif idx == "all":
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf_g(points_offset)
                sdf_grad = (
                    0.5
                    * (sdf_offset[..., 0::2, 0] - sdf_offset[..., 1::2, 0])
                    / eps
                )
            else:
                offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                    [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                ).to(points_unscaled)  # 3,3
                points_offset: Float[Tensor, "... 3 3"] = (
                    points_unscaled[..., None, :] + offsets
                ).clamp(-self.cfg.radius, self.cfg.radius)
                
                if isinstance(idx, int):
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf_i(points_offset, idx)
                elif isinstance(idx, list):
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf_edge(points_offset, idx)
                elif idx == "all":
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf_g(points_offset)
                
                sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
            normal = F.normalize(sdf_grad, dim=-1)
        
        else:
            raise AttributeError(f"Currently only support normal type 'finite_difference', got {self.cfg.normal_type}")
        
        return normal, sdf_grad
    
    def get_scaled_points(self, points_unscaled):
        points = contract_to_unisphere(
            points_unscaled, self.bbox, self.unbounded
        )
        return points
    
    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False, 
        curr_edge=None, no_global=False,
    ) -> Dict[str, Float[Tensor, "..."]]:
        
        grad_enabled = torch.is_grad_enabled()
        num_objects = self.cfg.num_objects if self.cfg.num_objects > 1 else 1
        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        # (1) get K object encodings:
        enc_OBJ = None # *N, K, 16
        for i in range(num_objects):
                            
            enc_i = self.encoding[i](points.view(-1, self.cfg.n_input_dims))  # *N, 16
            enc_out_dims = enc_i.shape[-1]
            enc_OBJ = enc_i.view(*points.shape[:-1], enc_out_dims)[..., None, :] if enc_OBJ is None else torch.cat(
                [enc_OBJ, 
                enc_i.view(*points.shape[:-1], enc_out_dims)[..., None, :]
                ], dim=-2)  # Nr, K, 16                

        sdf = self.sdf_network(enc_OBJ).view(*points.shape[:-1], num_objects)  # *N, K
        if self.cfg.sdf_bias == "learning":
            for i in range(num_objects):
                sdf[..., i:i+1] = self.get_shifted_sdf(points_unscaled, sdf[..., i:i+1], i)
        else:
            sdf = self.get_shifted_sdf(points_unscaled, sdf)            
        
        # normals and sdf_grads:
        normal_OBJ = None  # *N, 3, K
        sdf_grad_OBJ = None  # *N, 3, K
        for i in range(num_objects):
            normal_i, sdf_grad_i = self.get_normal(
                points_unscaled, enc_OBJ[..., i, :], sdf[..., i:i+1], i, grad_enabled)  # Nr, 3
            normal_OBJ = normal_i[..., None] if normal_OBJ is None else torch.cat([normal_OBJ, normal_i[..., None]], dim=-1)  # Nr, 3, K
            sdf_grad_OBJ = sdf_grad_i[..., None] if sdf_grad_OBJ is None else torch.cat([sdf_grad_OBJ, sdf_grad_i[..., None]], dim=-1)

        output = {
            "sdf_OBJ": sdf,  # [*N, K]
            "enc_OBJ": enc_OBJ,  # [*N, K, 16]
            "normal_OBJ": normal_OBJ,  # [*N, 3, K]
            "shading_normal_OBJ": normal_OBJ, 
            "sdf_grad_OBJ": sdf_grad_OBJ,
        }

        # features for color:
        if self.cfg.n_feature_dims > 0:
            features_OBJ = self.feature_network(enc_OBJ)  # [Nr, K, 3]
            output.update({"features_OBJ": features_OBJ, })  # [*N, K, 3]

        # (2) get global encodings:
        soft_Label = self.argmin_in_forward_softmin_in_backward(sdf)  # [*N, K]
        output.update({"soft_Label": soft_Label})
        
        if not no_global:
            
            normal_g = (soft_Label[..., None, :] * normal_OBJ).sum(dim=-1)
            sdf_grad_g = (soft_Label[..., None, :] * sdf_grad_OBJ).sum(dim=-1)
            
            if self.cfg.n_feature_dims > 0: 
                # [Nr, K, 1] * [Nr, K, 3]
                features_g = (soft_Label[..., None] * features_OBJ).sum(dim=-2)
                output.update({"features_g": features_g})
            
            output.update({"normal_g": normal_g, "shading_normal_g": normal_g, "sdf_grad_g": sdf_grad_g, })

    
        if curr_edge is not None:

            normal_edge = (soft_Label[..., None, :] * normal_OBJ)[..., curr_edge].sum(dim=-1)
            sdf_grad_edge = (soft_Label[..., None, :] * sdf_grad_OBJ)[..., curr_edge].sum(dim=-1)
            
            if self.cfg.n_feature_dims > 0: 
                # [Nr, K, 1] * [Nr, K, 3]
                features_edge = (soft_Label[..., None] * features_OBJ)[..., curr_edge, :].sum(dim=-2)
                output.update({"features_edge": features_edge})
            
            output.update({"normal_edge": normal_edge, "shading_normal_edge": normal_edge, "sdf_grad_edge": sdf_grad_edge, })

        return output
    
    def forward_sdf(self, 
        points: Float[Tensor, "*N Di"], return_enc=False
    ) -> Float[Tensor, "*N K"]:
        num_objects = self.cfg.num_objects
        points_unscaled = points  # [(Nr, 3), 3]
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        enc_OBJ = None
        for i in range(num_objects):
            enc_i = self.encoding[i](points.view(-1, self.cfg.n_input_dims))
            enc_out_dims = enc_i.shape[-1]
            
            enc_OBJ = enc_i.view(*points.shape[:-1], enc_out_dims)[..., None, :] if enc_OBJ is None else torch.cat(
                [
                    enc_OBJ, 
                    enc_i.view(*points.shape[:-1], enc_out_dims)[..., None, :]
                ], dim=-2)
        
        sdf = self.sdf_network(enc_OBJ).view(*points.shape[:-1], num_objects)
        if self.cfg.sdf_bias == "learning":
            for i in range(num_objects):
                sdf[..., i:i+1] = self.get_shifted_sdf(points_unscaled, sdf[..., i:i+1], i)
        else:
            sdf = self.get_shifted_sdf(points_unscaled, sdf)

        if return_enc:
            return enc_OBJ, sdf
        
        return sdf
    
    def forward_sdf_i(self, 
        points: Float[Tensor, "*N Di"], curr_idx: int
    ) -> Float[Tensor, "*N 1"]:
        
        num_objects = self.cfg.num_objects
        assert curr_idx < num_objects
        
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        enc_i = self.encoding[curr_idx](points.view(-1, self.cfg.n_input_dims))
        sdf_i = self.sdf_network(enc_i).view(*points.shape[:-1], 1)
        sdf_i = self.get_shifted_sdf(points_unscaled, sdf_i, curr_idx)
                
        return sdf_i
    
    def forward_sdf_edge(self, 
        points: Float[Tensor, "*N Di"], 
        curr_edge: List[int, ], 
    ) -> Float[Tensor, "*N 1"]:
        
        assert isinstance(curr_edge, list)

        points_unscaled = points
        enc_OBJ, sdf = self.forward_sdf(points_unscaled, True)
        
        soft_Label = self.argmin_in_forward_softmin_in_backward(sdf)
        enc_edge = (enc_OBJ * soft_Label[..., None])[..., curr_edge, :].sum(dim=-2)
        
        sdf_edge = self.sdf_network(enc_edge).view(*points.shape[:-1], 1)
        sdf_edge = self.get_shifted_sdf(points_unscaled, sdf_edge)
        
        return sdf_edge

    def forward_sdf_g(self, 
        points: Float[Tensor, "*N Di"]
    ) -> Float[Tensor, "*N 1"]:
        
        points_unscaled = points
        enc_OBJ, sdf = self.forward_sdf(points_unscaled, True)
        
        soft_Label = self.argmin_in_forward_softmin_in_backward(sdf)
        enc_g = (enc_OBJ * soft_Label[..., None]).sum(dim=-2)  # *N, 16

        sdf_g = self.sdf_network(enc_g).view(*points.shape[:-1], 1)
        sdf_g = self.get_shifted_sdf(points_unscaled, sdf_g)
        
        return sdf_g
    
    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N K"], Optional[Float[Tensor, "*N 3"]]]:
        
        num_objects = self.cfg.num_objects if self.cfg.num_objects > 1 else 1
        points_unscaled = points
        
        enc_OBJ, sdf = self.forward_sdf(points_unscaled, True)

        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.extract_mesh:
            if self.cfg.isosurface_deformable_grid:
                deformation = self.deformation_network(enc_OBJ).reshape(
                    *points.shape[:-1], num_objects, 3)
        return sdf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N K"]:
        return field - threshold

    def __export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        # enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        num_objects = self.cfg.num_objects
        enc_g = self.encoding[0](points.view(-1, self.cfg.n_input_dims))
        # enc_g = self.encoding_list.get_submodule(f"encoding_{0}")(points.view(-1, self.cfg.n_input_dims))
        for i in range(1, num_objects):
            enc_g += self.encoding[i](points.view(-1, self.cfg.n_input_dims))
            # enc_g += self.encoding_list.get_submodule(f"encoding_{i}")(points.view(-1, self.cfg.n_input_dims))
        
        features = self.feature_network(enc_g).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (
                    current_level - 1
                )
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    threestudio.info(
                        f"Update finite_difference_normal_eps to {grid_size}"
                    )
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(
                    f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}"
                )