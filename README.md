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
#### Will attempt to use most up to date packages available
#### I'm using Ubuntu 24.04 python3.12

```sh
git clone https://github.com/GGGHSL/GraphDreamer.git
cd GraphDreamer
```
Create environment:
```sh
python3 -m venv venv/GraphDreamer
source venv/GraphDreamer/bin/activate  # Repeat this step for every new terminal
```
Install dependencies:
```sh
pip install -r requirements.txt
```

Install [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for running Hash Grid based representations:
```sh
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install [NerfAcc](https://github.com/nerfstudio-project/nerfacc) for NeRF acceleration:
```sh
pip install git+https://github.com/KAIR-BAIR/nerfacc.git
```

Guidance model [DeepFloyd IF](https://github.com/deep-floyd/IF?tab=readme-ov-file) currently requires to accept its usage conditions. To do so, you need to have a [Hugging Face account](https://huggingface.co/welcome) (login in the terminal by `huggingface-cli login`) and accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0). 

## Quick Start
Generate a compositional scene of ```"a blue jay standing on a large basket of rainbow macarons"```:
```sh
bash scripts/blue_jay.sh
```
Results of the first (coarse) and the second (fine) stage will be save to ```examples/gd-if/blue_jay/``` and ```examples/gd-sd-refine/blue_jay/```. 

Try different seeds by setting ```seed=YOUR_SEED``` in the script. 
Use different tags to name different trials by setting ```export TG=YOUR_TAG``` to avoid overwriting. More examples can be found under ```scripts/```.

## Try with Your Own Prompts
Generating a compositional scene with GraphDreamer is as easy as with other dreamers. Here are the steps: 

#### Step 1 - Describe your objects
Give each object you want to create in the scene a prompt by setting
```sh
export P1=YOUR_TEXT_FOR_OBJECT_1
export P2=YOUR_TEXT_FOR_OBJECT_2
export P3=YOUR_TEXT_FOR_OBJECT_3
```
and ```system.prompt_obj=[["$P1"],["$P2"],["$P3"]]``` in the bash script .

By default, object SDFs will be initialized as spheres centered randomly, with the dispersion of the centers adjusted by multiplying a hyperparameter ```system.geometry.sdf_center_dispersion``` set to ```0.2```. 

#### Step 2 - Describe object relationships
Compose your objects into a scene by giving each object a prompt on its relationship to another object
```sh
export P12=RELATIONSHIP_BETWEEN_OBJECT_1_AND_2 
export P13=RELATIONSHIP_BETWEEN_OBJECT_1_AND_3
export P23=RELATIONSHIP_BETWEEN_OBJECT_2_AND_3
```
and add ```system.prompt_global=[["$P12"],["$P23"],["$P13"]]``` to your script. Based on these relationships, a graph is created accordingly with edges ```export E=[[0,1],[1,2],[0,2]]``` and ```system.edge_list=$E```.


Prompt the global scene by combining ```P12```, ```P13```, and ```P23``` into a sentence
```sh
export P=GLOBAL_TEXT_FOR_THE_SCENE
```
and add ```system.prompt_processor.prompt="$P"``` into the script. 




#### Step 3 - Negative prompts (optional)
In this compositional senarios, we found a simple way to create the "negative" prompt for individual objects. 
For each object, all other objects plus their relationships can be used as a negative prompt, 
```sh
export N1=$P23
export N2=$P13
export N3=$P12
```
and setting```system.prompt_obj_neg=[["$N1"],["$N2"],["$N3"]]```. 
You can further refine each negative prompts based on this general rule. 

#### Step 4 - Coarse-to-fine training
Start a new trainining simply by
```sh
export TG=YOUR_OWN_TAG
# Use different tags to avoid overwriting

python launch.py --config CONFIG_FILE --train --gpu 0 exp_root_dir="examples" system.geometry.num_objects=3 use_timestamp=false tag=$TG OTHER_CONFIGS
```
Set your own tag of the saving folder by ```export TG=YOUR_OWN_TAG``` and ```tag=$TG```, enable time stamps for naming the folder by setting```use_timestamp=true```.

The training configurations for the coarse stage are stored in ```configs/gd-if.yaml``` and the fine stage in ```configs/gd-sd-refine.yaml```. 

To resume from a previous checkpoint, e.g., resume from a coarse-stage training for the fine stage
```sh
resume=examples/gd-if/$TG/ckpts/last.ckpt
```


## More Applications
GraphDreamer can be used to inverse the semantics in a given image into a 3D scene, by extracting a scene graph directly from an input image with ChatGPT-4. 

To generate more objects and accelerate convergence, you may provide rough center coordinates for initializing each object by setting in the script:
```sh
export C=[[X1,Y1,Z1],[X2,Y2,Z3],...,[Xm,Ym,Zm]]
``` 
This will initialize the SDF-based objects as spheres centered at your given coordinates. The initial size of each object SDF sphere can also be custimized by setting the radius:
```sh
export R=[R1,R2,...,Rm]
```
Check ```./threestudio/models/geometry/gdreamer_implicit_sdf.py``` for more details on this implementation.

<!-- ## Code Structure
(TODO) -->

## Acknowledgement
The authors extend their thanks to Zehao Yu and Stefano Esposito for their invaluable feedback on the initial draft. Our thanks also go to Yao Feng, Zhen Liu, Zeju Qiu, Yandong Wen, and Yuliang Xiu for their proofreading of the final draft and for their insightful suggestions which enhanced the quality of this paper. Additionally, we appreciate the assistance of those who participated in our user study. 

Weiyang Liu and Bernhard Sch\"olkopf was supported by the German Federal Ministry of Education and Research (BMBF): T\"ubingen AI Center, FKZ: 01IS18039B, and by the Machine Learning Cluster of Excellence, the German Research Foundation (DFG): SFB 1233, Robust Vision: Inference Principles and Neural Mechanisms, TP XX, project number: 276693517. Andreas Geiger and Anpei Chen were supported by the ERC Starting Grant LEGO-3D (850533) and the DFG EXC number 2064/1 - project number 390727645.

This codebase is developed upon [threestudio](https://github.com/threestudio-project/threestudio). We appreciate its maintainers for their significant contributions to the community.


## Citation
```
@Inproceedings{gao2024graphdreamer,
  author    = {Gege Gao, Weiyang Liu, Anpei Chen, Andreas Geiger, Bernhard Schölkopf},
  title     = {GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
}
```
