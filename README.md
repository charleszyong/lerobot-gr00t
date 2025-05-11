## Grant Access

```shell
sudo chmod 666 /dev/ttyACM{0,1}
```

## Teleoperate

```shell
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
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

## LeRobot Framework

### Train a policy

```shell
python lerobot/scripts/train.py \
  --dataset.repo_id=charleyong/so100_ycube_put_box \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_so100_ycube_put_box \
  --job_name=pi0_so100_ycube_put_box \
  --policy.device=cuda \
  --wandb.enable=true
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
huggingface-cli download --repo-type dataset charleyong/so100_cubes_put_box \
--local-dir ./demo_data/so100_cubes_put_box
```

### Finetune

```shell 
python scripts/gr00t_finetune.py \
   --dataset-path ./demo_data/so100_ycube_put_box/ \
   --num-gpus 1 \
   --output-dir ./policy/so100-checkpoints  \
   --max-steps 10000 \
   --data-config so100 \
   --video-backend torchvision_av
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

Start the policy server:

```shell
python scripts/inference_service.py --server \
    --model_path /home/charles/Projects/lerobot/Isaac-GR00T/policy/so100-checkpoints/checkpoint-10000 \
    --embodiment_tag new_embodiment \
    --data_config so100 \
    --denoising_steps 4
```

Start the client node:

```shell
python getting_started/examples/eval_gr00t_so100.py \
 --use_policy --host 0.0.0.0 \
 --port 5555 
```
