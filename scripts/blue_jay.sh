export P="a DSLR photo of a blue jay standing on a large basket of rainbow macarons"
export P1="a DSLR photo of a blue jay"
export P2="a DSLR photo of a large basket of rainbow macarons"
export CD=0. 

# Use different tags to avoid overwriting:
export TG="blue_jay"

python launch.py --config configs/gd-if.yaml --train --gpu 0 exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=0. system.geometry.num_objects=2 system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions" system.prompt_obj=[["$P1"],["$P2"]] system.prompt_obj_neg=[["$P2"],["$P1"]] system.obj_use_view_dependent=true system.geometry.sdf_center_dispersion=$CD system.guidance.guidance_scale=[50.,20.] system.guidance.guidance_scale_milestones=[2000,] system.optimizer.params.geometry.lr=0.01

export RP=$P", 4K high-resolution high-quality"
export RP1=$P1", 4K high-resolution high-quality"
export RP2=$P2", 4K high-resolution high-quality"

python launch.py --config configs/gd-sd-refine.yaml --train --gpu 0 exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=0. system.geometry.num_objects=2 system.prompt_processor.prompt="$RP" system.prompt_obj=[["$RP1"],["$RP2"]] system.prompt_obj_neg=[["$P2"],["$P1"]] system.obj_use_view_dependent=true system.geometry.sdf_center_dispersion=$CD data.fovy_range=[70,90] data.eval_fovy_deg=90 resume=examples/gd-if/$TG/ckpts/last.ckpt  
# Adjust data.fovy_range to avoid OOM.