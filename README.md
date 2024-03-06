<!-- # GraphDreamer -->
# <p align="center">GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs </p>

###  <p align="center"> [Gege Gao](https://ggghsl.github.io/), [Weiyang Liu](https://wyliu.com/), [Anpei Chen](https://apchenstu.github.io/), [Andreas Geiger](https://www.cvlibs.net/), [Bernhard Schölkopf](https://is.mpg.de/~bs)</p>

###  <p align="center">CVPR 2024</p>

### <p align="center">[Full Paper](https://arxiv.org/pdf/2312.00093.pdf) | [arXiv](https://arxiv.org/abs/2312.00093) | [Project Page](https://graphdreamer.github.io/)

<p align="center">
  <img width="100%" src="assets/teaser.jpg"/>
</p><p align="center">
  <b>GraphDreamer</b> takes scene graphs as input and generates object compositional 3D scenes.
</p>

## Abstract
This repository contains a pytorch implementation for the paper [GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs](https://arxiv.org/abs/2312.00093). Our work present the first framework capable of generating **compositional 3D scenes** from **scene graphs**, where objects are represented as nodes and their interactions as edges. See the demo bellow to get a general idea.

<img width="100%" src="assets/demo.gif"/>


## Installation
#### Tested on CentOS 7.9 + Pytorch 2.0.1 
Create environment:
```sh
conda create -n GraphDreamer python=3.10
conda activate GraphDreamer
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for running Hash Grid based representations:
```sh
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install other dependencies:
```sh
pip install -r requirements.txt 
```

## Quick Start
Generate a compositional scene of "a blue jay standing on a large basket of rainbow macarons":
```sh
bash scripts/blue_jay.sh
```
Results of the first (coarse) and the second (fine) stage will be save to ```examples/gd-if/blue_jay``` and ```examples/gd-sd-refine/blue_jay```. 

Try different seeds by setting ```seed=YOUR_SEED``` in the script. 
Use different tags to name different trials by setting ```export TG=YOUR_TAG``` to avoid overwriting. 

## More applications
GraphDreamer can be used to inverse the semantics in a given image into a 3D scene, by extracting a scene graph directly from an input image with ChatGPT-4. 

To generate more objects and accelerate convergence, you may provide rough center coordinates for each object by setting in the script:
```sh
export C=[[X1,Y1,Z1],[X2,Y2,Z3],...,[Xm,Ym,Zm]]
``` 
This will initialize the SDF-based objects as spheres centered at your given coordinates. The initial size of each object SDF sphere can also be custimized by setting the radius:
```sh
export R=[R1,R2,...,Rm]
```
Check ```./threestudio/models/geometry/gdreamer_implicit_sdf.py``` for more details on this implementation.


## Acknowledgement
The authors extend their thanks to Zehao Yu and Stefano Esposito for their invaluable feedback on the initial draft. Our thanks also go to Yao Feng, Zhen Liu, Zeju Qiu, Yandong Wen, and Yuliang Xiu for their proofreading of the final draft and for their insightful suggestions which enhanced the quality of this paper. Additionally, we appreciate the assistance of those who participated in our user study. 

Weiyang Liu and Bernhard Sch\"olkopf was supported by the German Federal Ministry of Education and Research (BMBF): T\"ubingen AI Center, FKZ: 01IS18039B, and by the Machine Learning Cluster of Excellence, the German Research Foundation (DFG): SFB 1233, Robust Vision: Inference Principles and Neural Mechanisms, TP XX, project number: 276693517. Andreas Geiger and Anpei Chen were supported by the ERC Starting Grant LEGO-3D (850533) and the DFG EXC number 2064/1 - project number 390727645.

This codebase is developed upon [threestudio](https://github.com/threestudio-project/threestudio). We appreciate its maintainers for their significant contributions to the community.


## BibTex
```
@Inproceedings{gao2024graphdreamer,
  author    = {Gege Gao, Weiyang Liu, Anpei Chen, Andreas Geiger, Bernhard Schölkopf},
  title     = {GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
}
```
