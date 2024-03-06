import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.models.geometry.base import contract_to_unisphere
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("coordinate-background")
class CoordinateBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        dir_encoding_config: dict = field(
            default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3}
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        )
        random_aug: bool = False
        random_aug_prob: float = 0.5
        eval_color: Optional[Tuple[float, float, float]] = None
        # eval_color: Optional[Tuple[float, float, float]] = (1.0000, 0.7176, 0.6196)
        # eval_color: Optional[Tuple[float, float, float]] = (1.,1.,1.)

    cfg: Config

    # def configure(self) -> None:
    #     self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
    #     self.network = get_mlp(
    #         self.encoding.n_output_dims,
    #         self.cfg.n_output_dims,
    #         self.cfg.mlp_network_config,
    #     )
    #     self.encoding.requires_grad_(False)
    #     self.network.requires_grad_(False)
    #     # self.encoding = None
    #     # self.network = None

    def rand_aug_color(self, color: Float[Tensor, "*B 3"], w_color: Float = 0.5) -> Float[Tensor, "*B 3"]:
        color = get_activation(self.cfg.color_activation)(color)
        
        if (
            self.training
            and self.cfg.random_aug
            and random.random() < self.cfg.random_aug_prob
        ):  
            squeezed_dim = color.view(-1, 3).shape[0]
            color = color * w_color + (  # prevent checking for unused parameters in DDP
                torch.rand(self.cfg.n_output_dims)
                .to(color)[None, :]
                .expand(squeezed_dim, -1)
                .view(*color.shape[:-1], -1)
            ) * (1 - w_color)
            
        return color

    def normalize_points(
        self, 
        points_unscaled: Float[Tensor, "*N Di"], 
    ) -> Float[Tensor, "*N Di"]:
        
        points_scaled = contract_to_unisphere(
            points_unscaled, self.bbox, self.unbounded
        ) # points normalized to (0, 1)
        return points_scaled
    
    def forward(self, 
                norm_func, 
                dirs: Float[Tensor, "*B 3"],
                origins: Float[Tensor, "*B 3"],
                radius: Float) -> Float[Tensor, "*B 3"]:
        
        if not self.training and self.cfg.eval_color is not None:
            return torch.ones(*dirs.shape[:-1], self.cfg.n_output_dims).to(
                dirs
            ) * torch.as_tensor(self.cfg.eval_color).to(dirs)
        
        o_coord = origins * 0.  # 坐标原点
        d_1 = o_coord - origins
        d_1 = d_1.mul(dirs).sum(dim=-1, keepdim=True)  # 垂点与ray_o距离
        p_1 = origins + dirs * d_1  # 垂点坐标 [Nr, 3]
        
        d_2 = torch.sum(p_1 ** 2, dim=-1, keepdim=True)  # 垂点与坐标原点平方距离
        d_2 = (radius **2 - d_2) ** 0.5  # 沿着ray出方向，垂点与球面距离
        d = d_1 + d_2

        bg_coord = origins + dirs * d
        
        color = norm_func(bg_coord)
        # color = self.rand_aug_color(color, w_color=0.9)  # get activated, and check if need to do rand_aug
        color = self.rand_aug_color(color, w_color=0.0)

        return color
