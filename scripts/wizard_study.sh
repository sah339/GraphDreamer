export P="a Wizard standing before a Wooden Desk, gazing into a Crystal Ball perched atop the Wooden Desk, with a Stack of Ancient Spell Books perched atop the Wooden Desk, cartoon, blender"
export P1="'a Wizard: bearded, standing, focused'"
export P2="'a Wooden Desk: large, sturdy, carved with runes, aged'"
export P3="'a Crystal Ball with ornate stand: small, glowing, transparent, mystic'"
export P4="'a Stack of Ancient Spell Books: small, leather-bound, weathered, rune-etched'"
export NP="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

export P12="a Wizard standing before a Wooden Desk, cartoon, blender"
export P23="a Crystal Ball perched atop a Wooden Desk, cartoon, blender"
export P24="a Stack of Ancient Spell Books perched atop a Wooden Desk, cartoon, blender"
export P34="a Stack of Ancient Spell Books next to a Crystal Ball, cartoon, blender"
export N234="a Crystal Ball and a Stack of Ancient Spell Books perched atop a Wooden Desk"
export N124="a Wizard standing before an Wooden Desk, on which a Stack of Ancient Spell Books perched"
export N134="a Wizard intently gazing into a Crystal Ball, and a Stack of Ancient Spell Books"
export N123="a Wizard standing before an Wooden Desk, gazing into a Crystal Ball"

export PG=[["$P12"],["$P23"],["$P34"],["$P24"]]
export E=[[0,1],[1,2],[2,3],[1,3]]
export C=[[-0.25,0.1,0.],[0.24,0.12,0.],[0.25,0.13,0.2],[0.28,-0.16,0.2]]
# export C=[[-0.2487,0.0807,0.],[0.2445,0.1220,0.],[0.2555,0.1239,0.2],[0.2802,-0.1589,0.2]]
export R=[0.5,0.5,0.3,0.3]

# Name save folder:
export TG="wizard_study"

# 1. Coarse stage:
python launch.py --config configs/gd-if.yaml --train --gpu 0 exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=4 system.prompt_processor.prompt="$P" system.prompt_processor.negative_prompt="$NP" system.prompt_obj=[["$P1"],["$P2"],["$P3"],["$P4"]] system.prompt_obj_neg=[["$N234"],["$N134"],["$N124"],["$N123"]] system.prompt_global="$PG" system.edge_list=$E system.guidance.guidance_scale=[200.,100.] system.guidance.guidance_scale_milestones=[2000,] system.geometry.center_params=$C system.geometry.radius_params=$R system.optimizer.params.geometry.lr=0.01 data.resolution_milestones=[2000,] trainer.max_steps=4600

# 2. Fine stage:
export RP="a 4K DSLR high-resolution high-quality photo of "$P""
export RP1="'a 4K DSLR high-resolution high-quality photo of a Wizard: bearded, standing, focused'"
export RP2="'a 4K DSLR high-resolution high-quality photo of a Wooden Desk: large, sturdy, carved with runes, aged'"
export RP3="'a 4K DSLR high-resolution high-quality photo of a Crystal Ball with ornate stand: small, glowing, transparent, mystic'"
export RP4="'a 4K DSLR high-resolution high-quality photo of a Stack of Ancient Spell Books: small, leather-bound, weathered, rune-etched'"
export RP12="a 4K DSLR high-resolution high-quality photo of "$P12""
export RP23="a 4K DSLR high-resolution high-quality photo of "$P23""
export RP24="a 4K DSLR high-resolution high-quality photo of "$P24""
export RP34="a 4K DSLR high-resolution high-quality photo of "$P34""

export RPG=[["$RP12"],["$RP23"],["$RP34"],["$RP24"]]

# Avoid OOM: data.batch_size=1 data.width=128 data.height=128
python launch.py --config configs/gd-sd-refine.yaml --train --gpu 0 exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=4 system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]] system.prompt_obj_neg=[["$N234"],["$N134"],["$N124"],["$N123"]] system.prompt_global="$RPG" system.edge_list=$E system.geometry.center_params=$C system.geometry.radius_params=$R resume=examples/gd-if/$TG/ckpts/last.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=10000 trainer.val_check_interval=200

# Increase training resolution: data.width=256 data.height=256 (Optional: 1xA100 required)
python launch.py --config configs/gd-sd-refine.yaml --train --gpu 0 exp_root_dir="examples" use_timestamp=false tag=$TG system.loss.lambda_entropy=1. system.geometry.num_objects=4 system.prompt_processor.prompt="$RP" system.prompt_processor.negative_prompt="$NP" system.prompt_obj=[["$RP1"],["$RP2"],["$RP3"],["$RP4"]] system.prompt_obj_neg=[["$N234"],["$N134"],["$N124"],["$N123"]] system.prompt_global="$RPG" system.edge_list=$E system.geometry.center_params=$C system.geometry.radius_params=$R resume=examples/gd-sd-refine/$TG/ckpts/epoch=0-step=10000.ckpt data.batch_size=1 data.width=128 data.height=128 trainer.max_steps=20000 trainer.val_check_interval=200