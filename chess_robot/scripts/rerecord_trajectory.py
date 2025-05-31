#!/usr/bin/env python3
"""
Re-record a specific chess trajectory using teleoperation.
Useful when you need to redo a specific trajectory that didn't turn out well.
"""

import os
import sys
import time
import json
import numpy as np
import shutil
import threading
import torch
import argparse
from pathlib import Path
from datetime import datetime
from pynput import keyboard
from termcolor import colored

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.utils.utils import log_say

RECORDING_HZ = 30

class TrajectoryRerecorder:
    def __init__(self, robot: ManipulatorRobot):
        self.robot = robot
        self.recording = False
        self.current_trajectory = []
        self.current_timestamps = []
        self.keyboard_listener = None
        self.space_pressed = False
        self.abort_requested = False
        self.trajectories_dir = Path("chess_robot/trajectories")
        self.progress_file = Path("chess_robot/trajectories/progress.json")
        self.teleoperation_thread = None
        self.teleoperation_active = False
        self.latest_observation = None
        self.latest_action = None
        self.observation_lock = threading.Lock()
    
    def reset_flags(self):
        """Reset all control flags"""
        self.space_pressed = False
        self.abort_requested = False
        self.recording = False
        self.latest_observation = None
        self.latest_action = None
    
    def on_press(self, key):
        """Handle keyboard press events"""
        try:
            if key == keyboard.Key.space:
                self.space_pressed = True
            elif key == keyboard.Key.esc:
                self.abort_requested = True
                return False
        except AttributeError:
            pass
    
    def start_keyboard_listener(self):
        """Start listening for keyboard events"""
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()
    
    def stop_keyboard_listener(self):
        """Stop keyboard listener"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
    
    def display_status(self, square: str, action: str, status: str):
        """Display re-recording status"""
        os.system('clear')
        
        print(colored("╔════════════════════════════════════════╗", "cyan"))
        print(colored("║   Chess Trajectory Re-Recording        ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        print(colored(f"║  Square: {square:<29} ║", "cyan"))
        print(colored(f"║  Action: {action.upper():<29} ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        
        if self.teleoperation_active:
            print(colored("║  🎮 TELEOPERATION ACTIVE 🎮            ║", "green", attrs=['bold']))
        else:
            print(colored("║  Teleoperation: OFF                    ║", "white"))
        
        print(colored("║                                        ║", "cyan"))
        print(colored(f"║  Status: {status:<29} ║", "yellow" if "Position" in status else "green"))
        print(colored("║                                        ║", "cyan"))
        print(colored("║  SPACE - Start/Stop recording          ║", "white"))
        print(colored("║  ESC - Abort                          ║", "white"))
        print(colored("╚════════════════════════════════════════╝", "cyan"))
    
    def teleoperation_loop(self):
        """Run teleoperation with data recording"""
        try:
            while self.teleoperation_active and not self.abort_requested:
                if self.recording:
                    # When recording, use teleop_step with record_data=True
                    obs_dict, action_dict = self.robot.teleop_step(record_data=True)
                    
                    # Store both observation and action
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
    
    def rerecord_trajectory(self, square: str, action: str, backup: bool = True) -> bool:
        """Re-record a specific trajectory"""
        # Check if trajectory exists
        filepath = self.trajectories_dir / action / f"{square}.npz"
        if not filepath.exists():
            print(colored(f"Trajectory {square} {action} doesn't exist!", "red"))
            return False
        
        # Create backup if requested
        if backup:
            backup_dir = self.trajectories_dir / "backups" / action
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{square}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            shutil.copy2(filepath, backup_path)
            print(colored(f"Created backup: {backup_path}", "green"))
        
        self.current_trajectory = []
        self.current_timestamps = []
        self.space_pressed = False
        self.abort_requested = False
        self.latest_observation = None
        self.latest_action = None
        
        # Start teleoperation
        self.start_teleoperation()
        
        # Wait for user to position robot
        while not self.space_pressed and not self.abort_requested:
            self.display_status(square, action, "Position leader arm and press SPACE")
            time.sleep(0.1)
        
        if self.abort_requested:
            self.stop_teleoperation()
            return False
        
        # Start recording
        self.space_pressed = False
        self.recording = True
        start_time = time.time()
        log_say(f"Re-recording {action} for {square}", play_sounds=True)
        
        # Record trajectory
        while not self.space_pressed and not self.abort_requested:
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
            
            # Update status
            duration = time.time() - start_time
            self.display_status(square, action, f"Recording... ({duration:.1f}s)")
            time.sleep(0.05)
        
        self.recording = False
        self.stop_teleoperation()
        
        if self.abort_requested:
            print(colored("Recording aborted!", "red"))
            return False
        
        # Save trajectory
        if len(self.current_trajectory) > 5:
            positions = np.array(self.current_trajectory)
            timestamps = np.array(self.current_timestamps)
            
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
                recorded_with='teleoperation',
                rerecorded=True
            )
            
            log_say(f"Successfully re-recorded {action} for {square}", play_sounds=True)
            print(colored(f"Saved: {filepath}", "green"))
            print(f"Duration: {timestamps[-1]:.2f}s, Frames: {len(positions)}")
            return True
        else:
            print(colored("Trajectory too short! Aborting.", "red"))
            return False
    
    def list_available_trajectories(self):
        """List all recorded trajectories"""
        trajectories = []
        for action in ['pickup', 'putdown']:
            action_dir = self.trajectories_dir / action
            if action_dir.exists():
                for file in sorted(action_dir.glob("*.npz")):
                    square = file.stem
                    trajectories.append((square, action))
        return trajectories
    
    def select_trajectory(self):
        """Interactive trajectory selection"""
        trajectories = self.list_available_trajectories()
        
        if not trajectories:
            print(colored("No trajectories found!", "red"))
            return None, None
        
        print(colored("\n=== Available Trajectories ===", "cyan"))
        print(colored("Format: [index] square action", "white"))
        print()
        
        # Group by action for better display
        pickup_trajs = [(s, a) for s, a in trajectories if a == 'pickup']
        putdown_trajs = [(s, a) for s, a in trajectories if a == 'putdown']
        
        print(colored("PICKUP trajectories:", "yellow"))
        for i, (square, action) in enumerate(pickup_trajs):
            if i % 8 == 0 and i > 0:
                print()
            print(f"[{i:2d}] {square}", end="  ")
        print("\n")
        
        print(colored("PUTDOWN trajectories:", "yellow"))
        offset = len(pickup_trajs)
        for i, (square, action) in enumerate(putdown_trajs):
            if i % 8 == 0 and i > 0:
                print()
            print(f"[{offset + i:2d}] {square}", end="  ")
        print("\n")
        
        # Get user selection
        while True:
            try:
                choice = input(colored("Enter trajectory index (or 'q' to quit): ", "green"))
                if choice.lower() == 'q':
                    return None, None
                
                idx = int(choice)
                if 0 <= idx < len(trajectories):
                    return trajectories[idx]
                else:
                    print(colored("Invalid index! Please try again.", "red"))
            except ValueError:
                # Try parsing as square + action
                parts = choice.strip().split()
                if len(parts) == 2:
                    square, action = parts
                    if (square, action) in trajectories:
                        return square, action
                print(colored("Invalid input! Enter index or 'square action' (e.g., 'e4 pickup')", "red"))
    
    def load_trajectory(self, square: str, action: str):
        """Load existing trajectory data"""
        filepath = self.trajectories_dir / action / f"{square}.npz"
        if filepath.exists():
            data = np.load(filepath)
            return dict(data)
        return None
    
    def display_trajectory_info(self, square: str, action: str, data: dict):
        """Display information about existing trajectory"""
        print(colored(f"\n=== Trajectory Info: {square} {action} ===", "cyan"))
        print(f"Recorded on: {data.get('date', 'Unknown')}")
        print(f"Number of frames: {data.get('num_frames', len(data['positions']))}")
        print(f"Duration: {data['timestamps'][-1]:.2f} seconds")
        print(f"Frequency: {data.get('frequency', RECORDING_HZ)} Hz")
        print(f"Recording method: {data.get('recorded_with', 'manual')}")
    
    def run(self):
        """Main re-recording workflow"""
        self.start_keyboard_listener()
        
        # Initial message about teleoperation
        print(colored("\n🎮 TELEOPERATION MODE 🎮", "cyan", attrs=['bold']))
        print("You will control the robot using the leader arm (left side).")
        print("The follower arm (right side) will mimic your movements and be recorded.")
        print()
        
        try:
            while True:
                try:
                    # Reset all flags for new recording
                    self.reset_flags()
                    
                    # Select trajectory
                    square, action = self.select_trajectory()
                    if square is None:
                        break
                    
                    # Load and display existing trajectory
                    existing_data = self.load_trajectory(square, action)
                    if existing_data:
                        self.display_trajectory_info(square, action, existing_data)
                    else:
                        print(colored(f"Warning: No existing trajectory found for {square} {action}", "yellow"))
                    
                    # Temporarily stop keyboard listener for user input
                    self.stop_keyboard_listener()
                    
                    # Confirm re-recording
                    print()
                    confirm = input(colored("Re-record this trajectory? [y/N]: ", "yellow"))
                    if confirm.lower() != 'y':
                        self.start_keyboard_listener()
                        continue
                    
                    # Restart keyboard listener
                    self.start_keyboard_listener()
                    
                    # Record new trajectory
                    print(colored("\nPreparing to record...", "green"))
                    time.sleep(1)
                    
                    success = self.rerecord_trajectory(square, action)
                    if success:
                        # Temporarily stop keyboard listener for user input
                        self.stop_keyboard_listener()
                        
                        # Ask if user wants to continue
                        print()
                        another = input(colored("Re-record another trajectory? [y/N]: ", "green"))
                        
                        # Restart keyboard listener
                        self.start_keyboard_listener()
                        
                        if another.lower() != 'y':
                            print(colored("Exiting re-recording mode.", "cyan"))
                            break
                        else:
                            print(colored("\nContinuing to next trajectory...\n", "cyan"))
                            # Small delay before continuing
                            time.sleep(0.5)
                            continue
                    else:
                        print(colored("Recording cancelled.", "yellow"))
                        break
                        
                except KeyboardInterrupt:
                    print(colored("\nInterrupted by user.", "yellow"))
                    break
                except Exception as e:
                    print(colored(f"\nError in main loop: {e}", "red"))
                    import traceback
                    traceback.print_exc()
                    print()
                    
                    # Stop keyboard listener for input
                    self.stop_keyboard_listener()
                    retry = input(colored("Continue despite error? [y/N]: ", "yellow"))
                    self.start_keyboard_listener()
                    
                    if retry.lower() != 'y':
                        break
                    
        finally:
            self.stop_teleoperation()
            self.stop_keyboard_listener()
            print(colored("\nCleaning up...", "cyan"))


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
        
        # Create re-recorder and run
        rerecorder = TrajectoryRerecorder(robot)
        rerecorder.run()
        
    except Exception as e:
        print(colored(f"\nFatal error: {e}", "red"))
        import traceback
        traceback.print_exc()
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main() 