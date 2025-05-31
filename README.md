## Grant Access

```shell
sudo chmod 666 /dev/ttyACM{0,1}
```

## Teleoperate

```shell
python lerobot/scripts/control_robot.py \
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

## Chess Robot Trajectory System

This section describes the chess robot trajectory recording and execution system for the SO100 robot.

### Overview

The chess robot system allows recording and playing back trajectories for chess piece movements using **teleoperation**. It consists of:
- Recording 128 trajectories (64 squares × 2 actions: pickup/putdown) using the leader-follower arm setup
- Re-recording specific trajectories to fix mistakes
- Executing trajectories with adjustable speed
- Managing trajectory data with utilities

### 1. Recording All Trajectories (Teleoperation)

Record all 128 chess trajectories using the SO100's teleoperation feature:

```shell
python chess_robot/scripts/record_chess_trajectories.py
```

**How it works:**
- The SO100 robot has two arms: a leader arm (left) and a follower arm (right)
- You manually move the **leader arm** to control the robot
- The **follower arm** mimics your movements in real-time
- The follower arm's positions are recorded for playback

**Features:**
- 🎮 **Teleoperation mode**: Control the robot naturally by moving the leader arm
- Records trajectories for all 64 squares (a1-h8)
- Records both pickup and putdown actions for each square
- Press SPACE to start/stop recording for each trajectory
- Press S to skip a trajectory
- Press Q to save progress and quit
- Press ESC for emergency stop
- **Automatically resumes from where you left off** if interrupted

The recording process:
1. The robot enters teleoperation mode (follower mimics leader)
2. Move the leader arm to the starting position
3. Press SPACE to begin recording
4. Move the leader arm to perform the action (pickup or putdown)
5. The follower arm follows your movements and gets recorded
6. Press SPACE to stop recording
7. The trajectory is automatically saved

### 2. Re-recording Specific Trajectories (Teleoperation)

If you make a mistake or want to improve a specific trajectory:

```shell
python chess_robot/scripts/rerecord_trajectory.py
```

**Features:**
- Uses the same teleoperation mode as initial recording
- Interactive menu to select which trajectory to re-record
- Shows information about the existing trajectory
- Creates a backup of the original trajectory
- Can re-record multiple trajectories in one session

You can select trajectories by:
- Index number from the displayed list
- Square and action (e.g., "e4 pickup")

### 3. Executing Trajectories

Execute recorded trajectories with speed control (no teleoperation needed for playback):

```shell
# Interactive mode (menu-driven)
python chess_robot/scripts/execute_trajectory.py

# Execute a single trajectory
python chess_robot/scripts/execute_trajectory.py --square e4 --action pickup --speed 1.5

# Execute a complete chess move
python chess_robot/scripts/execute_trajectory.py --move e2 e4 --speed 0.8

# Preview trajectory without execution
python chess_robot/scripts/execute_trajectory.py --square e4 --action pickup --preview
```

**Options:**
- `--square`: Chess square (e.g., e4)
- `--action`: Action type (pickup or putdown)
- `--speed`: Speed factor (0.1-2.0, default=1.0)
- `--move FROM TO`: Execute a complete move (pickup from FROM, putdown at TO)
- `--preview`: Show trajectory plot without executing
- `--hz`: Execution frequency in Hz (default=30)

**During execution:**
- Press SPACE to pause/resume
- Press ESC to abort

### 4. Managing Trajectories

Utility script for trajectory management:

```shell
# Check recording progress
python chess_robot/scripts/trajectory_utils.py progress

# Verify all trajectory files
python chess_robot/scripts/trajectory_utils.py verify

# Create a backup
python chess_robot/scripts/trajectory_utils.py backup
python chess_robot/scripts/trajectory_utils.py backup --name my_backup_name

# Restore from backup
python chess_robot/scripts/trajectory_utils.py restore --name my_backup_name

# Clear progress (keeps trajectory files)
python chess_robot/scripts/trajectory_utils.py clear

# Clear everything (DANGEROUS - deletes all files)
python chess_robot/scripts/trajectory_utils.py clear --delete-files

# Export trajectory metadata to CSV
python chess_robot/scripts/trajectory_utils.py export
```

### File Structure

```
chess_robot/
├── trajectories/
│   ├── pickup/
│   │   ├── a1.npz
│   │   ├── a2.npz
│   │   └── ... (64 files)
│   ├── putdown/
│   │   ├── a1.npz
│   │   ├── a2.npz
│   │   └── ... (64 files)
│   └── progress.json
├── backups/
│   └── backup_YYYYMMDD_HHMMSS/
└── scripts/
    ├── record_chess_trajectories.py
    ├── rerecord_trajectory.py
    ├── execute_trajectory.py
    └── trajectory_utils.py
```

### Tips for Recording with Teleoperation

1. **Leader Arm Control**: Move the leader arm (left side) smoothly and naturally
2. **Follower Observation**: Watch the follower arm (right side) to ensure it's tracking correctly
3. **Consistent Speed**: Try to maintain consistent movement speed during recording
4. **Starting Position**: Ensure both arms are in a comfortable starting position before recording
5. **Gripper Sync**: The gripper on both arms should be synchronized
6. **Smooth Movements**: Avoid jerky movements as they will be replicated in playback
7. **Test First**: Do a few practice runs without recording to get comfortable with teleoperation

### Teleoperation Advantages

- **Natural Movement**: Move the robot as you would naturally move your own arm
- **Precise Control**: Direct physical feedback helps achieve precise positioning
- **Quick Recording**: Faster than programming waypoints manually
- **Intuitive**: No need to think about joint angles or coordinates
- **Consistent Results**: The follower arm ensures consistent force and grip

### Integration with Chess Engine

To integrate with a chess engine:

```python
from chess_robot.scripts.execute_trajectory import TrajectoryExecutor
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.configs import So100RobotConfig

# Initialize robot
config = So100RobotConfig()
robot = ManipulatorRobot(config)
robot.connect()

# Create executor
executor = TrajectoryExecutor(robot)

# Execute a chess move
executor.execute_move("e2", "e4", speed_factor=1.2)

# Disconnect
robot.disconnect()
```

### Troubleshooting

**"Trajectory too short" error**: Make sure to record for at least 0.5 seconds

**Teleoperation not working**: Ensure both leader and follower arms are properly connected and calibrated

**Arms out of sync**: Check that both arms start from the same position before recording

**"Port is in use" error**: The servo disable script should handle this, but if issues persist, use:
```shell
python lerobot/scripts/disable_servos.py
```

**Missing dependencies**: Install required packages:
```shell
pip install pynput termcolor scipy matplotlib
```
