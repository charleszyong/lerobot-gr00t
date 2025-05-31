## Grant Access

```shell
sudo chmod 666 /dev/ttyACM{0,1}
```

## Teleoperate

```shell
python lerobot/scripts/control_robot.py \                                                    ─╯
  --robot.type=so100 \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

## Record a dataset

```shell
HF_USER=$(huggingface-cli whoami | head -n1) && echo "$HF_USER"

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the orange cubes and place them in the brown box." \
  --control.repo_id=${HF_USER}/so100_cubes_put_box \
  --control.tags='["so100","yc_demo"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=25 \
  --control.reset_time_s=5 \
  --control.num_episodes=40 \
  --control.push_to_hub=true \
  --control.display_data=true
```

Multi-command Dataset

```shell
HF_USER=$(huggingface-cli whoami | head -n1) && echo "$HF_USER"
DATASET_ID=${HF_USER}/so100_tic_tac_toe_autoend && echo "$DATASET_ID"

INSTRUCTIONS=(
  "Place the circle to the center left box"
  "Place the circle to the center right box"
  "Place the circle to the center box"
  "Place the circle to the center top box"
  "Place the circle to the center bottom box"
  "Place the circle to the bottom left corner box"
  "Place the circle to the bottom right corner box"
  "Place the circle to the top left corner box"
  "Place the circle to the top right corner box"
)

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="${INSTRUCTIONS[9]}" \
  --control.repo_id="${DATASET_ID}" \
  --control.tags='["so100","auto_end","multi_task", "new_office"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=20 \
  --control.push_to_hub=true \
  --control.display_data=true \
  --control.resume=true  # **key line – keeps adding to the same dataset**
```

## LeRobot Framework

### Train a policy

```shell
python lerobot/scripts/train.py \
  --dataset.repo_id="${DATASET_ID}" \
  --policy.type=act \
  --output_dir=outputs/train/act_so100_tic_tac_toe_autoend \
  --job_name=act_so100_tic_tac_toe_autoend \
  --policy.device=cuda \
  --wandb.enable=true \
  --batch_size=64 \
  --num_workers=16 
```

### Run a policy

```shell
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the orange cube marked Y and place it in the brown box." \
  --control.repo_id=${HF_USER}/eval_diffusion_so100_ycube_put_box \
  --control.tags='["eval","yc_demo"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=10 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/diffusion_so100_ycube_put_box/checkpoints/last/pretrained_model
```

## GR00T Framework

### Download Dataset

```shell
huggingface-cli download --repo-type dataset "${DATASET_ID}" \
--local-dir ./demo_data/so100_tic_tac_toe_autoend
```

### Finetune

```shell 
python scripts/gr00t_finetune.py \
   --dataset-path ./demo_data/so100_tic_tac_toe_autoend/ \
   --num-gpus 1 \
   --output-dir ./policy/so100_tic_tac_toe_autoend_extended-checkpoints  \
   --max-steps 100000 \
   --data-config so100 \
   --video-backend torchvision_av \
   --save-steps 25000 \
   --batch-size 64 
```

### Open-loop Evaluation 

```shell
python scripts/eval_policy.py --plot \
   --embodiment_tag new_embodiment \
   --model_path /home/charles/Projects/lerobot/Isaac-GR00T/policy/so100-checkpoints/checkpoint-10000 \
   --data_config so100 \
  --dataset_path /home/charles/.cache/huggingface/lerobot/charleyong/so100_ycube_put_box/ \
   --video_backend torchvision_av \
   --modality_keys single_arm gripper \
   --steps 900
```

### Run a policy

Download Policy

```shell
rsync -avhP lambda:/home/ubuntu/lerobot-gr00t/Isaac-GR00T/demo_data/LegrandFrederic/lego-pickup-dual-setup-gr00t/  /home/charles/Projects/lerobot/Isaac-GR00T/policy/lego-pickup-dual-setup-gr00t
```

Start the policy server:

```shell
python scripts/inference_service.py --server \
    --model_path /home/charles/Projects/lerobot/Isaac-GR00T/policy/so100_tic_tac_toe_autoend-checkpoints/checkpoint-7000 \
    --embodiment_tag new_embodiment \
    --data_config so100 \
    --denoising_steps 8
```

Start the client node:

```shell
python getting_started/examples/eval_gr00t_so100.py \
 --use_policy --host 0.0.0.0 \
 --port 5555 \
 --lang_instruction="Place the circle to the center box" \
 --cam_idx 0
```
