#!/usr/bin/env python3
"""
Execute recorded chess trajectories with speed control.
"""

import os
import sys
import time
import numpy as np
import torch
import argparse
from pathlib import Path
from pynput import keyboard
from termcolor import colored
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.utils.utils import log_say

class TrajectoryExecutor:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.trajectories_dir = Path("chess_robot/trajectories")
        self.keyboard_listener = None
        self.abort_requested = False
        self.pause_requested = False
        
    def on_press(self, key):
        """Handle keyboard press events"""
        try:
            if key == keyboard.Key.esc:
                self.abort_requested = True
                return False
            elif key == keyboard.Key.space:
                self.pause_requested = not self.pause_requested
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
    
    def load_trajectory(self, square: str, action: str):
        """Load trajectory data"""
        filepath = self.trajectories_dir / action / f"{square}.npz"
        if not filepath.exists():
            raise FileNotFoundError(f"Trajectory not found: {filepath}")
        
        data = np.load(filepath)
        return dict(data)
    
    def interpolate_trajectory(self, positions, timestamps, speed_factor=1.0):
        """Interpolate trajectory for smooth execution with speed control"""
        # Adjust timestamps for speed factor
        adjusted_timestamps = timestamps / speed_factor
        
        # Create interpolation functions for each joint
        num_joints = positions.shape[1]
        interpolators = []
        
        for joint in range(num_joints):
            # Use cubic interpolation for smooth motion
            interp_func = interp1d(
                adjusted_timestamps, 
                positions[:, joint], 
                kind='cubic',
                bounds_error=False,
                fill_value=(positions[0, joint], positions[-1, joint])
            )
            interpolators.append(interp_func)
        
        return interpolators, adjusted_timestamps
    
    def display_execution_status(self, square: str, action: str, progress: float, speed: float, paused: bool = False):
        """Display execution status"""
        os.system('clear')
        
        progress_bar = '█' * int(progress * 20) + '░' * (20 - int(progress * 20))
        
        print(colored("╔════════════════════════════════════════╗", "cyan"))
        print(colored("║     Chess Trajectory Execution         ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        print(colored(f"║  Square: {square:<29} ║", "cyan"))
        print(colored(f"║  Action: {action.upper():<29} ║", "cyan"))
        print(colored(f"║  Speed: {speed:.1f}x                              ║", "cyan"))
        print(colored("║                                        ║", "cyan"))
        print(colored(f"║  [{progress_bar}] {progress*100:3.0f}%         ║", "cyan"))
        
        if paused:
            print(colored("║  Status: PAUSED                        ║", "yellow"))
        else:
            print(colored("║  Status: EXECUTING                     ║", "green"))
            
        print(colored("║                                        ║", "cyan"))
        print(colored("║  SPACE - Pause/Resume                  ║", "white"))
        print(colored("║  ESC - Abort execution                 ║", "white"))
        print(colored("╚════════════════════════════════════════╝", "cyan"))
    
    def execute_trajectory(self, square: str, action: str, speed_factor: float = 1.0, 
                         preview: bool = False, execution_hz: float = 30.0):
        """Execute a trajectory with given speed"""
        # Load trajectory
        data = self.load_trajectory(square, action)
        positions = data['positions']
        timestamps = data['timestamps']
        
        print(colored(f"\nLoaded trajectory: {square} {action}", "green"))
        print(f"Duration: {timestamps[-1]:.2f}s (will execute in {timestamps[-1]/speed_factor:.2f}s)")
        print(f"Frames: {len(positions)}")
        
        if preview:
            self.preview_trajectory(positions, timestamps, square, action)
            return
        
        # Interpolate trajectory
        interpolators, adjusted_timestamps = self.interpolate_trajectory(
            positions, timestamps, speed_factor
        )
        
        # Reset flags
        self.abort_requested = False
        self.pause_requested = False
        
        # Execute trajectory
        start_time = time.time()
        execution_period = 1.0 / execution_hz
        
        log_say(f"Executing {action} for {square}", play_sounds=True)
        
        while True:
            if self.abort_requested:
                print(colored("\nExecution aborted!", "red"))
                break
            
            if self.pause_requested:
                self.display_execution_status(square, action, progress, speed_factor, paused=True)
                time.sleep(0.1)
                continue
            
            # Calculate current time in trajectory
            current_time = (time.time() - start_time)
            
            # Check if trajectory is complete
            if current_time >= adjusted_timestamps[-1]:
                progress = 1.0
                self.display_execution_status(square, action, progress, speed_factor)
                print(colored("\nTrajectory completed!", "green"))
                break
            
            # Interpolate current position
            current_position = []
            for interp_func in interpolators:
                current_position.append(interp_func(current_time))
            
            # Send to robot
            action_tensor = torch.tensor(current_position, dtype=torch.float32)
            self.robot.send_action(action_tensor)
            
            # Update display
            progress = current_time / adjusted_timestamps[-1]
            self.display_execution_status(square, action, progress, speed_factor)
            
            # Sleep to maintain execution frequency
            time.sleep(execution_period)
    
    def preview_trajectory(self, positions, timestamps, square, action):
        """Preview trajectory with matplotlib"""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'Trajectory Preview: {square} {action}')
        
        joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow Flex', 
                      'Wrist Flex', 'Wrist Roll', 'Gripper']
        
        for i, (ax, name) in enumerate(zip(axes.flat, joint_names)):
            ax.plot(timestamps, positions[:, i])
            ax.set_title(name)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def execute_move(self, from_square: str, to_square: str, speed_factor: float = 1.0):
        """Execute a complete chess move (pickup from one square, putdown to another)"""
        print(colored(f"\nExecuting move: {from_square} → {to_square}", "cyan", attrs=['bold']))
        
        # Execute pickup
        self.execute_trajectory(from_square, 'pickup', speed_factor)
        if self.abort_requested:
            return
        
        # Small delay between actions
        time.sleep(0.5)
        
        # Execute putdown
        self.execute_trajectory(to_square, 'putdown', speed_factor)
    
    def interactive_mode(self):
        """Interactive trajectory execution"""
        trajectories = self.list_available_trajectories()
        
        if not trajectories:
            print(colored("No trajectories found!", "red"))
            return
        
        while True:
            print(colored("\n=== Chess Trajectory Executor ===", "cyan", attrs=['bold']))
            print("\nOptions:")
            print("1. Execute single trajectory")
            print("2. Execute chess move (pickup + putdown)")
            print("3. List all trajectories")
            print("4. Quit")
            
            choice = input(colored("\nEnter choice (1-4): ", "green"))
            
            if choice == '1':
                # Single trajectory execution
                square = input("Enter square (e.g., e4): ").strip().lower()
                action = input("Enter action (pickup/putdown): ").strip().lower()
                
                if action not in ['pickup', 'putdown']:
                    print(colored("Invalid action! Use 'pickup' or 'putdown'", "red"))
                    continue
                
                try:
                    speed = float(input("Enter speed factor (0.1-2.0, default=1.0): ") or "1.0")
                    speed = max(0.1, min(2.0, speed))
                    
                    preview = input("Preview trajectory? [y/N]: ").lower() == 'y'
                    
                    self.execute_trajectory(square, action, speed, preview)
                    
                except FileNotFoundError:
                    print(colored(f"Trajectory not found for {square} {action}!", "red"))
                except Exception as e:
                    print(colored(f"Error: {e}", "red"))
                    
            elif choice == '2':
                # Chess move execution
                from_square = input("Enter from square (e.g., e2): ").strip().lower()
                to_square = input("Enter to square (e.g., e4): ").strip().lower()
                
                try:
                    speed = float(input("Enter speed factor (0.1-2.0, default=1.0): ") or "1.0")
                    speed = max(0.1, min(2.0, speed))
                    
                    self.execute_move(from_square, to_square, speed)
                    
                except FileNotFoundError as e:
                    print(colored(f"Missing trajectory: {e}", "red"))
                except Exception as e:
                    print(colored(f"Error: {e}", "red"))
                    
            elif choice == '3':
                # List trajectories
                print(colored("\n=== Available Trajectories ===", "cyan"))
                pickup_trajs = [(s, a) for s, a in trajectories if a == 'pickup']
                putdown_trajs = [(s, a) for s, a in trajectories if a == 'putdown']
                
                print(colored("\nPICKUP trajectories:", "yellow"))
                for i, (square, _) in enumerate(pickup_trajs):
                    if i % 8 == 0 and i > 0:
                        print()
                    print(f"{square}", end="  ")
                
                print(colored("\n\nPUTDOWN trajectories:", "yellow"))
                for i, (square, _) in enumerate(putdown_trajs):
                    if i % 8 == 0 and i > 0:
                        print()
                    print(f"{square}", end="  ")
                print()
                
            elif choice == '4':
                break
            else:
                print(colored("Invalid choice!", "red"))


def main():
    parser = argparse.ArgumentParser(description="Execute chess trajectories")
    parser.add_argument('--square', type=str, help='Chess square (e.g., e4)')
    parser.add_argument('--action', type=str, choices=['pickup', 'putdown'], 
                       help='Action type')
    parser.add_argument('--speed', type=float, default=1.0, 
                       help='Speed factor (0.1-2.0, default=1.0)')
    parser.add_argument('--move', type=str, nargs=2, metavar=('FROM', 'TO'),
                       help='Execute chess move (e.g., --move e2 e4)')
    parser.add_argument('--preview', action='store_true', 
                       help='Preview trajectory without execution')
    parser.add_argument('--hz', type=float, default=30.0,
                       help='Execution frequency in Hz (default=30)')
    
    args = parser.parse_args()
    
    # Initialize robot
    print("Initializing SO100 robot...")
    config = So100RobotConfig()
    robot = ManipulatorRobot(config)
    
    try:
        robot.connect()
        print("Robot connected successfully!")
        
        # Create executor
        executor = TrajectoryExecutor(robot)
        executor.start_keyboard_listener()
        
        try:
            if args.move:
                # Execute a chess move
                executor.execute_move(args.move[0], args.move[1], args.speed)
            elif args.square and args.action:
                # Execute single trajectory
                executor.execute_trajectory(
                    args.square, args.action, args.speed, 
                    args.preview, args.hz
                )
            else:
                # Interactive mode
                executor.interactive_mode()
        finally:
            executor.stop_keyboard_listener()
            
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main() 