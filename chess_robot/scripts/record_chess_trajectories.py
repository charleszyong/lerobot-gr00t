#!/usr/bin/env python3
"""
Record chess piece trajectories for all 64 squares using teleoperation.
The user moves the leader arm while the follower arm mimics and gets recorded.
Supports resuming from where you left off after interruption.
"""

import os
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from pynput import keyboard
import threading
from termcolor import colored

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.utils.utils import log_say

# Chess board configuration
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']
ACTIONS = ['pickup', 'putdown']
RECORDING_HZ = 30

class ChessTrajectoryRecorder:
    def __init__(self, robot: ManipulatorRobot):
        self.robot = robot
        self.recording = False
        self.current_trajectory = []
        self.current_timestamps = []
        self.keyboard_listener = None
        self.space_pressed = False
        self.skip_requested = False
        self.quit_requested = False
        self.progress_file = Path("chess_robot/trajectories/progress.json")
        self.trajectories_dir = Path("chess_robot/trajectories")
        self.teleoperation_thread = None
        self.teleoperation_active = False
        self.latest_observation = None
        self.latest_action = None
        self.observation_lock = threading.Lock()
        self.keyboard_listener_active = False
        
        # Load progress if exists
        self.progress = self.load_progress()
        
    def load_progress(self):
        """Load recording progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"completed": [], "last_square": None, "last_action": None}
    
    def save_progress(self):
        """Save current progress"""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_trajectory_path(self, square: str, action: str) -> Path:
        """Get the file path for a trajectory"""
        return self.trajectories_dir / action / f"{square}.npz"
    
    def is_trajectory_recorded(self, square: str, action: str) -> bool:
        """Check if a trajectory has already been recorded"""
        return f"{square}_{action}" in self.progress["completed"]
    
    def mark_trajectory_completed(self, square: str, action: str):
        """Mark a trajectory as completed"""
        key = f"{square}_{action}"
        if key not in self.progress["completed"]:
            self.progress["completed"].append(key)
        self.progress["last_square"] = square
        self.progress["last_action"] = action
        self.save_progress()
    
    def on_press(self, key):
        """Handle keyboard press events"""
        try:
            if key == keyboard.Key.space:
                self.space_pressed = True
            elif key == keyboard.Key.esc:
                self.quit_requested = True
                return False
            elif hasattr(key, 'char'):
                if key.char == 's':
                    self.skip_requested = True
                elif key.char == 'q':
                    self.quit_requested = True
                    return False
        except AttributeError:
            pass
    
    def start_keyboard_listener(self):
        """Start listening for keyboard events"""
        if not self.keyboard_listener_active:
            try:
                self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
                self.keyboard_listener.start()
                self.keyboard_listener_active = True
            except Exception as e:
                print(colored(f"Warning: Could not start keyboard listener: {e}", "yellow"))
                self.keyboard_listener_active = False
    
    def stop_keyboard_listener(self):
        """Stop keyboard listener"""
        if self.keyboard_listener_active and self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
                self.keyboard_listener_active = False
            except Exception as e:
                # Ignore errors when stopping, the listener might already be stopped
                self.keyboard_listener_active = False
    
    def display_status(self, square: str, action: str, status: str, completed: int, total: int):
        """Display recording status"""
        os.system('clear')
        progress_pct = (completed / total) * 100
        progress_bar = '█' * int(progress_pct / 5) + '░' * (20 - int(progress_pct / 5))
        
        print(colored("╔════════════════════════════════════════╗", "cyan"))
        print(colored("║   Chess Trajectory Recording (Teleop)  ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        print(colored(f"║  Current Square: {square:<21} ║", "cyan"))
        print(colored(f"║  Action: {action.upper():<29} ║", "cyan"))
        print(colored(f"║  Progress: {completed}/{total} ({progress_pct:.0f}%)              ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        print(colored(f"║  [{progress_bar}] {progress_pct:3.0f}%         ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        
        if self.teleoperation_active:
            print(colored("║  🎮 TELEOPERATION ACTIVE 🎮            ║", "green", attrs=['bold']))
        else:
            print(colored("║  Teleoperation: OFF                    ║", "white"))
        
        print(colored("║                                        ║", "cyan"))
        print(colored(f"║  Status: {status:<29} ║", "yellow" if "Wait" in status or "Position" in status else "green"))
        print(colored("║                                        ║", "cyan"))
        print(colored("║  Commands:                             ║", "cyan"))
        print(colored("║  SPACE - Start/Stop recording          ║", "white"))
        print(colored("║  S - Skip this trajectory              ║", "white"))
        print(colored("║  Q - Save and quit                     ║", "white"))
        print(colored("║  ESC - Emergency stop                  ║", "white"))
        print(colored("╚════════════════════════════════════════╝", "cyan"))
    
    def teleoperation_loop(self):
        """Run teleoperation with data recording"""
        try:
            while self.teleoperation_active and not self.quit_requested:
                # Use the robot's built-in teleoperation with data recording
                if self.recording:
                    # When recording, use teleop_step with record_data=True
                    obs_dict, action_dict = self.robot.teleop_step(record_data=True)
                    
                    # Store both observation and action for the recording thread
                    with self.observation_lock:
                        self.latest_observation = obs_dict
                        self.latest_action = action_dict
                else:
                    # When not recording, just run teleoperation
                    self.robot.teleop_step(record_data=False)
                
                # Sleep to maintain control frequency
                time.sleep(1.0 / RECORDING_HZ)
        except Exception as e:
            print(colored(f"Teleoperation error: {e}", "red"))
    
    def start_teleoperation(self):
        """Start teleoperation mode"""
        if not self.teleoperation_active:
            self.teleoperation_active = True
            self.teleoperation_thread = threading.Thread(target=self.teleoperation_loop)
            self.teleoperation_thread.start()
    
    def stop_teleoperation(self):
        """Stop teleoperation mode"""
        if self.teleoperation_active:
            self.teleoperation_active = False
            if self.teleoperation_thread:
                self.teleoperation_thread.join()
    
    def record_trajectory(self, square: str, action: str) -> bool:
        """Record a single trajectory using teleoperation"""
        self.current_trajectory = []
        self.current_timestamps = []
        self.space_pressed = False
        self.skip_requested = False
        self.latest_observation = None
        self.latest_action = None
        
        # Start teleoperation
        self.start_teleoperation()
        
        # Wait for user to position robot using leader arm
        while not self.space_pressed and not self.skip_requested and not self.quit_requested:
            status = f"Move leader arm to start position"
            completed = len(self.progress["completed"])
            self.display_status(square, action, status, completed, 128)
            time.sleep(0.1)
        
        if self.skip_requested:
            self.stop_teleoperation()
            return True
        if self.quit_requested:
            self.stop_teleoperation()
            return False
        
        # Start recording
        self.space_pressed = False
        self.recording = True
        start_time = time.time()
        log_say(f"Recording {action} for {square}", play_sounds=True)
        
        # Record at specified frequency
        while not self.space_pressed and not self.quit_requested:
            # Get the latest observation and action from teleoperation thread
            with self.observation_lock:
                if self.latest_observation is not None and self.latest_action is not None:
                    # Get follower positions (joints 0-4) and leader gripper position (joint 5)
                    follower_state = self.latest_observation["observation.state"].numpy()
                    leader_action = self.latest_action["action"].numpy()
                    
                    # Create mixed state: follower positions for joints 0-4, leader position for gripper (joint 5)
                    mixed_state = follower_state.copy()
                    mixed_state[5] = leader_action[5]  # Replace gripper with leader's gripper position
                    
                    self.current_trajectory.append(mixed_state)
                    self.current_timestamps.append(time.time() - start_time)
            
            # Display recording status
            duration = time.time() - start_time
            status = f"Recording... ({duration:.1f}s)"
            completed = len(self.progress["completed"])
            self.display_status(square, action, status, completed, 128)
            
            # Sleep to maintain display update frequency
            time.sleep(0.05)  # Update display at 20Hz
        
        self.recording = False
        self.stop_teleoperation()
        
        if self.quit_requested:
            return False
        
        # Save trajectory
        if len(self.current_trajectory) > 5:  # Minimum trajectory length
            self.save_trajectory(square, action)
            log_say(f"Saved {action} trajectory for {square}", play_sounds=True)
            return True
        else:
            print(colored("Trajectory too short! Please record again.", "red"))
            time.sleep(2)
            return self.record_trajectory(square, action)
    
    def save_trajectory(self, square: str, action: str):
        """Save recorded trajectory to file"""
        filepath = self.get_trajectory_path(square, action)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy arrays
        positions = np.array(self.current_trajectory)
        timestamps = np.array(self.current_timestamps)
        
        # Save data
        np.savez(
            filepath,
            positions=positions,
            timestamps=timestamps,
            frequency=RECORDING_HZ,
            square=square,
            action=action,
            robot_type='so100',
            date=datetime.now().isoformat(),
            num_frames=len(positions),
            recorded_with='teleoperation'
        )
        
        # Mark as completed
        self.mark_trajectory_completed(square, action)
    
    def find_next_trajectory(self):
        """Find the next trajectory to record"""
        for rank in RANKS:
            for file in FILES:
                square = file + rank
                for action in ACTIONS:
                    if not self.is_trajectory_recorded(square, action):
                        return square, action
        return None, None
    
    def run(self):
        """Main recording loop"""
        self.start_keyboard_listener()
        
        try:
            # Check if resuming
            if self.progress["completed"]:
                print(colored(f"Resuming from previous session. {len(self.progress['completed'])}/128 trajectories completed.", "green"))
                time.sleep(2)
            
            # Initial message about teleoperation
            print(colored("\n🎮 TELEOPERATION MODE 🎮", "cyan", attrs=['bold']))
            print("You will control the robot using the leader arm (left side).")
            print("The follower arm (right side) will mimic your movements and be recorded.")
            print("\nPress any key to continue...")
            input()
            
            while True:
                # Find next trajectory to record
                square, action = self.find_next_trajectory()
                
                if square is None:
                    print(colored("All trajectories recorded! 🎉", "green"))
                    break
                
                # Record trajectory
                success = self.record_trajectory(square, action)
                
                if not success:
                    print(colored("Recording stopped by user.", "yellow"))
                    break
                
                # Small delay between recordings
                time.sleep(0.5)
                
        finally:
            self.stop_teleoperation()
            self.stop_keyboard_listener()
            print(f"\nRecorded {len(self.progress['completed'])}/128 trajectories.")
            print(f"Progress saved to {self.progress_file}")


def main():
    # Initialize robot
    print("Initializing SO100 robot...")
    config = So100RobotConfig()
    robot = ManipulatorRobot(config)
    
    try:
        robot.connect()
        print("Robot connected successfully!")
        
        # Verify robot has leader and follower arms
        if not hasattr(robot, 'leader_arms') or not hasattr(robot, 'follower_arms'):
            print(colored("Error: This robot doesn't support teleoperation!", "red"))
            return
        
        # Create recorder and run
        recorder = ChessTrajectoryRecorder(robot)
        recorder.run()
        
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main() 