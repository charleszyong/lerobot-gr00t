#!/usr/bin/env python3
"""
Re-record a specific chess trajectory.
Useful for fixing mistakes in individual recordings.
"""

import os
import sys
import time
import json
import numpy as np
import shutil
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

class TrajectoryReRecorder:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.recording = False
        self.current_trajectory = []
        self.current_timestamps = []
        self.keyboard_listener = None
        self.space_pressed = False
        self.quit_requested = False
        self.trajectories_dir = Path("chess_robot/trajectories")
        self.progress_file = Path("chess_robot/trajectories/progress.json")
    
    def on_press(self, key):
        """Handle keyboard press events"""
        try:
            if key == keyboard.Key.space:
                self.space_pressed = True
            elif key == keyboard.Key.esc:
                self.quit_requested = True
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
    
    def display_recording_status(self, square: str, action: str, status: str, recording: bool = False):
        """Display recording status"""
        os.system('clear')
        
        print(colored("╔════════════════════════════════════════╗", "cyan"))
        print(colored("║     Re-Record Chess Trajectory         ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        print(colored(f"║  Square: {square:<29} ║", "cyan"))
        print(colored(f"║  Action: {action.upper():<29} ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        
        color = "green" if recording else "yellow"
        print(colored(f"║  Status: {status:<29} ║", color))
        
        print(colored("║                                        ║", "cyan"))
        print(colored("║  SPACE - Start/Stop recording          ║", "white"))
        print(colored("║  ESC - Cancel and exit                 ║", "white"))
        print(colored("╚════════════════════════════════════════╝", "cyan"))
    
    def record_trajectory(self, square: str, action: str) -> bool:
        """Record a new trajectory"""
        self.current_trajectory = []
        self.current_timestamps = []
        self.space_pressed = False
        
        # Wait for user to position robot
        while not self.space_pressed and not self.quit_requested:
            status = f"Position robot and press SPACE"
            self.display_recording_status(square, action, status)
            time.sleep(0.1)
        
        if self.quit_requested:
            return False
        
        # Start recording
        self.space_pressed = False
        self.recording = True
        start_time = time.time()
        log_say(f"Recording {action} for {square}", play_sounds=True)
        
        # Record at specified frequency
        while not self.space_pressed and not self.quit_requested:
            # Capture robot state
            observation = self.robot.capture_observation()
            state = observation["observation.state"]
            
            self.current_trajectory.append(state.numpy())
            self.current_timestamps.append(time.time() - start_time)
            
            # Display recording status
            duration = time.time() - start_time
            status = f"Recording... ({duration:.1f}s)"
            self.display_recording_status(square, action, status, recording=True)
            
            # Sleep to maintain frequency
            time.sleep(1.0 / RECORDING_HZ)
        
        self.recording = False
        
        if self.quit_requested:
            return False
        
        # Check trajectory length
        if len(self.current_trajectory) > 5:
            return True
        else:
            print(colored("\nTrajectory too short! Please record again.", "red"))
            time.sleep(2)
            return self.record_trajectory(square, action)
    
    def save_trajectory(self, square: str, action: str, backup_original: bool = True):
        """Save recorded trajectory to file"""
        filepath = self.trajectories_dir / action / f"{square}.npz"
        
        # Backup original if it exists
        if backup_original and filepath.exists():
            backup_path = filepath.with_suffix('.npz.backup')
            shutil.copy(filepath, backup_path)
            print(colored(f"Original trajectory backed up to {backup_path}", "green"))
        
        # Ensure directory exists
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
            num_frames=len(positions)
        )
        
        print(colored(f"Trajectory saved to {filepath}", "green"))
    
    def run(self):
        """Main re-recording workflow"""
        self.start_keyboard_listener()
        
        try:
            while True:
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
                
                # Confirm re-recording
                print()
                confirm = input(colored("Re-record this trajectory? [y/N]: ", "yellow"))
                if confirm.lower() != 'y':
                    continue
                
                # Record new trajectory
                print(colored("\nPreparing to record...", "green"))
                time.sleep(1)
                
                success = self.record_trajectory(square, action)
                if success:
                    # Save new trajectory
                    self.save_trajectory(square, action)
                    log_say(f"Successfully re-recorded {action} for {square}", play_sounds=True)
                    
                    # Ask if user wants to continue
                    print()
                    another = input(colored("Re-record another trajectory? [y/N]: ", "green"))
                    if another.lower() != 'y':
                        break
                else:
                    print(colored("Recording cancelled.", "yellow"))
                    break
                    
        finally:
            self.stop_keyboard_listener()


def main():
    # Initialize robot
    print("Initializing SO100 robot...")
    config = So100RobotConfig()
    robot = ManipulatorRobot(config)
    
    try:
        robot.connect()
        print("Robot connected successfully!")
        
        # Create re-recorder and run
        rerecorder = TrajectoryReRecorder(robot)
        rerecorder.run()
        
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main() 