import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("diffuse-with-point-light-material")
class DiffuseWithPointLightMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        ambient_light_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
        diffuse_light_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)
        diffuse_prob: float = 0.75
        textureless_prob: float = 0.5
        albedo_activation: str = "sigmoid"  # scale_-11_01 | lambda x: x * 0.5 + 0.5
        soft_shading: bool = False

        '''Ref-NeuS: '''
        input_feature_dims: int = 8
        color_activation: str = "sigmoid"
        dir_encoding_config: dict = field(
            default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3}
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 128,
                "n_hidden_layers": 4,
            }
        )

        rgb_network_config: dict = field(
            default_factory=lambda: {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        use_ref_neus: bool = False

    cfg: Config
    requires_normal: bool = True

    def configure(self) -> None:
        self.ambient_light_color: Float[Tensor, "3"]
        self.register_buffer(
            "ambient_light_color",
            torch.as_tensor(self.cfg.ambient_light_color, dtype=torch.float32),
        )
        self.diffuse_light_color: Float[Tensor, "3"]
        self.register_buffer(
            "diffuse_light_color",
            torch.as_tensor(self.cfg.diffuse_light_color, dtype=torch.float32),
        )
        self.ambient_only = False
        
        if self.cfg.use_ref_neus:
            '''Ref-NeuS: '''
            self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
            self.n_input_dims = self.cfg.input_feature_dims + self.encoding.n_output_dims + 3  # 23 = 18 + points + normal
            self.network = get_mlp(self.n_input_dims, 3, self.cfg.mlp_network_config)

            self.rgb_net = get_mlp(self.cfg.input_feature_dims, 3, self.cfg.rgb_network_config)

    def forward(
        self,
        features: Float[Tensor, "B ... Nf"],
        positions: Float[Tensor, "B ... 3"],
        shading_normal: Float[Tensor, "B ... 3"],
        light_positions: Float[Tensor, "B ... 3"],
        ambient_ratio: Optional[float] = None,
        shading: Optional[str] = None,
        **kwargs,
    ) -> Float[Tensor, "B ... 3"]: 
        if features.shape[-1] != 3:
            
            features = self.rgb_net(features).view(*features.shape[:-1], 3)
        
        albedo = get_activation(self.cfg.albedo_activation)(features[..., :3])  # scale_-11_01 | lambda x: x * 0.5 + 0.5

        if ambient_ratio is not None:
            # if ambient ratio is specified, use it
            diffuse_light_color = (1 - ambient_ratio) * torch.ones_like(
                self.diffuse_light_color
            )
            ambient_light_color = ambient_ratio * torch.ones_like(
                self.ambient_light_color
            )
        elif self.training and self.cfg.soft_shading:
            # otherwise if in training and soft shading is enabled, random a ambient ratio
            diffuse_light_color = torch.full_like(
                self.diffuse_light_color, random.random()
            )
            ambient_light_color = 1.0 - diffuse_light_color
        else:
            # otherwise use the default fixed values
            diffuse_light_color = self.diffuse_light_color  # (0.9, 0.9, 0.9)
            ambient_light_color = self.ambient_light_color  # (0.1, 0.1, 0.1)

        light_directions: Float[Tensor, "B ... 3"] = F.normalize(
            light_positions - positions, dim=-1
        )
        diffuse_light: Float[Tensor, "B ... 3"] = (
            dot(shading_normal, light_directions).clamp(min=0.0) * diffuse_light_color
        )
        textureless_color = diffuse_light + ambient_light_color
        # clamp albedo to [0, 1] to compute shading
        color = albedo.clamp(0.0, 1.0) * textureless_color

        if shading is None:
            if self.training:
                # adopt the same type of augmentation for the whole batch
                if self.ambient_only or random.random() > self.cfg.diffuse_prob:  # 0.75
                    shading = "albedo"
                elif random.random() < self.cfg.textureless_prob:  # 0.5
                    shading = "textureless"
                else:
                    shading = "diffuse"
            else:
                if self.ambient_only:
                    shading = "albedo"
                else:
                    # return shaded color by default in evaluation
                    shading = "diffuse"

        # multiply by 0 to prevent checking for unused parameters in DDP
        if shading == "albedo":
            return albedo + textureless_color * 0
        elif shading == "textureless":
            return albedo * 0 + textureless_color
        elif shading == "diffuse":
            return color
        else:
            raise ValueError(f"Unknown shading type {shading}")
    
    def forward_refneus(
        self,
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        positions: Float[Tensor, "B ... 3"],
        shading_normal: Float[Tensor, "B ... 3"],
        # light_positions: Float[Tensor, "B ... 3"],
        # ambient_ratio: Optional[float] = None,
        # shading: Optional[str] = None,
        **kwargs,
    ) -> Float[Tensor, "B ... K 3"]:
        # viewdirs and normals must be normalized before passing to this function
        viewdirs = (viewdirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        viewdirs_embd = self.encoding(viewdirs.view(-1, 3))

        network_inp = torch.cat(
            [
                positions, 
                viewdirs_embd,
                shading_normal, 
                features.view(-1, features.shape[-1])], dim=-1
        )
        
        color = self.network(network_inp.view(-1, network_inp.shape[-1]))
        color = color.view(*features.shape[:-1], 3)
        color = get_activation(self.cfg.color_activation)(color)

        return color
    
    def forward_objects(
        self,
        features: Float[Tensor, "B ... Nf"],
        # viewdirs: Float[Tensor, "*B 3"],
        positions: Float[Tensor, "B ... 3"],
        shading_normal: Float[Tensor, "B ... 3 K"],
        light_positions: Float[Tensor, "B ... 3"],
        ambient_ratio: Optional[float] = None,
        shading: Optional[str] = None,
        **kwargs,
    ) -> Float[Tensor, "B ... K 3"]: 
        
        K = shading_normal.shape[-1]
        albedo = get_activation(self.cfg.albedo_activation)(features[..., :3])  # scale_-11_01 | lambda x: x * 0.5 + 0.5
        # print("albedo:", albedo.shape)  # Nr, K, 3

        if ambient_ratio is not None:
            # if ambient ratio is specified, use it
            diffuse_light_color = (1 - ambient_ratio) * torch.ones_like(
                self.diffuse_light_color
            )
            ambient_light_color = ambient_ratio * torch.ones_like(
                self.ambient_light_color
            )
        elif self.training and self.cfg.soft_shading:
            # otherwise if in training and soft shading is enabled, random a ambient ratio
            diffuse_light_color = torch.full_like(
                self.diffuse_light_color, random.random()
            )
            ambient_light_color = 1.0 - diffuse_light_color
        else:
            # otherwise use the default fixed values
            diffuse_light_color = self.diffuse_light_color  # (0.9, 0.9, 0.9)
            ambient_light_color = self.ambient_light_color  # (0.1, 0.1, 0.1)

        light_directions: Float[Tensor, "B ... 3"] = F.normalize(
            light_positions - positions, dim=-1
        )
        
        # print(albedo.dim(), shading_normal.dim(), shading_normal.shape)
        if albedo.dim() < shading_normal.dim(): # Nr, 3
            albedo = albedo[..., None, :].expand(*albedo.shape[:-1], K, 3)
        
        color = albedo.clamp(0.0, 1.0)
    
        textureless_color = torch.zeros_like(color)
        for i in range(K):
            diffuse_light: Float[Tensor, "B ... 3"] = (
                dot(shading_normal[..., i], light_directions).clamp(min=0.0) * diffuse_light_color
            )
            _textureless_color = diffuse_light + ambient_light_color
            # clamp albedo to [0, 1] to compute shading
            color[..., i, :] *= _textureless_color
            textureless_color[..., i, :] = _textureless_color
        
        if shading is None:
            if self.training:
                # adopt the same type of augmentation for the whole batch
                if self.ambient_only or random.random() > self.cfg.diffuse_prob:  # 0.75
                    shading = "albedo"
                elif random.random() < self.cfg.textureless_prob:  # 0.5
                    shading = "textureless"
                else:
                    shading = "diffuse"
            else:
                if self.ambient_only:
                    shading = "albedo"
                else:
                    # return shaded color by default in evaluation
                    shading = "diffuse"
        
        # multiply by 0 to prevent checking for unused parameters in DDP
        if shading == "albedo":
            return albedo.clamp(0.0, 1.0) + textureless_color * 0
        elif shading == "textureless":
            return albedo.clamp(0.0, 1.0) * 0 + textureless_color
        elif shading == "diffuse":
            return color
        else:
            raise ValueError(f"Unknown shading type {shading}")
    
    def forward_objects_refneus(
        self,
        points: Float[Tensor, "*N Di"],
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        # positions: Float[Tensor, "B ... 3"],
        shading_normal: Float[Tensor, "B ... 3 K"],
        # light_positions: Float[Tensor, "B ... 3"],
        # ambient_ratio: Optional[float] = None,
        # shading: Optional[str] = None,
        **kwargs,
    ) -> Float[Tensor, "B ... K 3"]:
        # viewdirs and normals must be normalized before passing to this function
        viewdirs = (viewdirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        viewdirs_embd = self.encoding(viewdirs.view(-1, 3))

        K = shading_normal.shape[-1]
        color = torch.zeros((*features.shape[:-1], K, 3)).to(features)  # B ... K 3
        network_inp = torch.zeros((*features.shape[:-1], K, self.n_input_dims)).to(features)

        for i in range(K):
            shading_normal_i = shading_normal[..., i]
            
            if features.dim() < shading_normal.dim():
                feature_i = features.view(-1, features.shape[-1])
            else:
                # [..., K, 3]
                feature_i = features[..., i, :].view(-1, features.shape[-1])

            network_inp_i = torch.cat(
                [points, 
                 viewdirs_embd,
                 shading_normal_i, 
                 feature_i], dim=-1
            )
            # print(features.shape[-1])
            # print(self.n_input_dims)
            # print(network_inp.shape)
            network_inp[..., i, :] += network_inp_i
        
            # color_i = self.network(network_inp_i)
            # color_i = color_i.view(*features.shape[:-1], 3)
            # color_i = get_activation(self.cfg.color_activation)(color_i)
            # color[..., i, :] += color_i
        
        color = self.network(network_inp.view(-1, network_inp.shape[-1]))
        color = color.view(*features.shape[:-1], K, 3)
        color = get_activation(self.cfg.color_activation)(color)

        return color
    
    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.ambient_only = True

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        albedo = get_activation(self.cfg.albedo_activation)(features[..., :3]).clamp(
            0.0, 1.0
        )
        return {"albedo": albedo}
