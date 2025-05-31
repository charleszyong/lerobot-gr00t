#!/usr/bin/env python3
"""
Utility script for managing chess trajectories.
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from termcolor import colored

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class TrajectoryManager:
    def __init__(self):
        self.trajectories_dir = Path("chess_robot/trajectories")
        self.progress_file = self.trajectories_dir / "progress.json"
        
    def load_progress(self):
        """Load recording progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"completed": [], "last_square": None, "last_action": None}
    
    def show_progress(self):
        """Display recording progress"""
        progress = self.load_progress()
        completed = len(progress["completed"])
        total = 128  # 64 squares * 2 actions
        
        print(colored("=== Chess Trajectory Recording Progress ===", "cyan", attrs=['bold']))
        print(f"\nCompleted: {completed}/{total} ({completed/total*100:.1f}%)")
        
        if progress["last_square"] and progress["last_action"]:
            print(f"Last recorded: {progress['last_square']} {progress['last_action']}")
        
        # Create grid view
        print(colored("\n=== Grid View ===", "cyan"))
        print("Legend: ✓ = both recorded, P = pickup only, D = putdown only, · = not recorded")
        print()
        
        # Chess board display
        print("   ", end="")
        for file in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            print(f" {file} ", end="")
        print()
        
        for rank in ['8', '7', '6', '5', '4', '3', '2', '1']:
            print(f" {rank} ", end="")
            for file in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                square = file + rank
                pickup_done = f"{square}_pickup" in progress["completed"]
                putdown_done = f"{square}_putdown" in progress["completed"]
                
                if pickup_done and putdown_done:
                    print(colored(" ✓ ", "green"), end="")
                elif pickup_done:
                    print(colored(" P ", "yellow"), end="")
                elif putdown_done:
                    print(colored(" D ", "yellow"), end="")
                else:
                    print(" · ", end="")
            print()
        
        # List missing trajectories
        print(colored("\n=== Missing Trajectories ===", "red"))
        missing = []
        for rank in ['1', '2', '3', '4', '5', '6', '7', '8']:
            for file in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                square = file + rank
                for action in ['pickup', 'putdown']:
                    if f"{square}_{action}" not in progress["completed"]:
                        missing.append(f"{square} {action}")
        
        if missing:
            print(f"Total missing: {len(missing)}")
            print("First 10 missing:")
            for i, traj in enumerate(missing[:10]):
                print(f"  - {traj}")
        else:
            print(colored("All trajectories recorded! 🎉", "green"))
    
    def verify_trajectories(self):
        """Verify all trajectory files"""
        print(colored("=== Verifying Trajectory Files ===", "cyan", attrs=['bold']))
        
        issues = []
        total_size = 0
        trajectory_count = 0
        
        for action in ['pickup', 'putdown']:
            action_dir = self.trajectories_dir / action
            if not action_dir.exists():
                continue
                
            for file in action_dir.glob("*.npz"):
                trajectory_count += 1
                total_size += file.stat().st_size
                
                try:
                    data = np.load(file)
                    positions = data['positions']
                    timestamps = data['timestamps']
                    
                    # Check for issues
                    if len(positions) < 10:
                        issues.append(f"{file.name}: Very short trajectory ({len(positions)} frames)")
                    
                    if timestamps[-1] < 0.5:
                        issues.append(f"{file.name}: Very quick trajectory ({timestamps[-1]:.2f}s)")
                    
                    if timestamps[-1] > 10:
                        issues.append(f"{file.name}: Very long trajectory ({timestamps[-1]:.2f}s)")
                        
                except Exception as e:
                    issues.append(f"{file.name}: Error loading file - {e}")
        
        print(f"\nTotal trajectories: {trajectory_count}")
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
        
        if issues:
            print(colored(f"\nFound {len(issues)} potential issues:", "yellow"))
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(colored("\nAll trajectories look good!", "green"))
    
    def backup_trajectories(self, backup_name: str = None):
        """Create a backup of all trajectories"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = Path("chess_robot/backups") / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(colored(f"Creating backup: {backup_dir}", "cyan"))
        
        # Copy trajectories
        if self.trajectories_dir.exists():
            shutil.copytree(self.trajectories_dir, backup_dir / "trajectories", dirs_exist_ok=True)
            print("✓ Trajectories backed up")
        
        print(colored(f"\nBackup complete: {backup_dir}", "green"))
    
    def restore_backup(self, backup_name: str):
        """Restore trajectories from backup"""
        backup_dir = Path("chess_robot/backups") / backup_name
        
        if not backup_dir.exists():
            print(colored(f"Backup not found: {backup_dir}", "red"))
            return
        
        print(colored(f"Restoring from backup: {backup_dir}", "cyan"))
        
        # Backup current state first
        if self.trajectories_dir.exists():
            temp_backup = f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.backup_trajectories(temp_backup)
        
        # Restore trajectories
        src_traj = backup_dir / "trajectories"
        if src_traj.exists():
            if self.trajectories_dir.exists():
                shutil.rmtree(self.trajectories_dir)
            shutil.copytree(src_traj, self.trajectories_dir)
            print("✓ Trajectories restored")
        
        print(colored("\nRestore complete!", "green"))
    
    def clear_progress(self, keep_files: bool = True):
        """Clear recording progress"""
        if not keep_files:
            # Dangerous: removes all trajectory files
            confirm = input(colored("WARNING: This will DELETE all trajectory files! Type 'DELETE' to confirm: ", "red"))
            if confirm != "DELETE":
                print("Cancelled.")
                return
            
            if self.trajectories_dir.exists():
                shutil.rmtree(self.trajectories_dir)
            print(colored("All trajectory files deleted!", "red"))
        else:
            # Just clear progress tracking
            if self.progress_file.exists():
                os.remove(self.progress_file)
            print(colored("Progress tracking cleared. Trajectory files kept.", "yellow"))
    
    def export_metadata(self):
        """Export trajectory metadata to CSV"""
        import csv
        
        output_file = self.trajectories_dir / "trajectory_metadata.csv"
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Square', 'Action', 'Duration (s)', 'Frames', 'File Size (KB)', 'Date Recorded'])
            
            for action in ['pickup', 'putdown']:
                action_dir = self.trajectories_dir / action
                if not action_dir.exists():
                    continue
                    
                for file in sorted(action_dir.glob("*.npz")):
                    try:
                        data = np.load(file)
                        square = file.stem
                        duration = data['timestamps'][-1]
                        frames = len(data['positions'])
                        size_kb = file.stat().st_size / 1024
                        date = data.get('date', 'Unknown')
                        
                        writer.writerow([square, action, f"{duration:.2f}", frames, f"{size_kb:.1f}", date])
                    except Exception as e:
                        writer.writerow([file.stem, action, "ERROR", "ERROR", "ERROR", str(e)])
        
        print(colored(f"Metadata exported to: {output_file}", "green"))


def main():
    parser = argparse.ArgumentParser(description="Chess trajectory management utilities")
    parser.add_argument('command', choices=['progress', 'verify', 'backup', 'restore', 'clear', 'export'],
                       help='Command to execute')
    parser.add_argument('--name', type=str, help='Backup name (for backup/restore)')
    parser.add_argument('--delete-files', action='store_true', 
                       help='Delete trajectory files when clearing (DANGEROUS!)')
    
    args = parser.parse_args()
    
    manager = TrajectoryManager()
    
    if args.command == 'progress':
        manager.show_progress()
    elif args.command == 'verify':
        manager.verify_trajectories()
    elif args.command == 'backup':
        manager.backup_trajectories(args.name)
    elif args.command == 'restore':
        if not args.name:
            print(colored("Error: --name required for restore", "red"))
            print("\nAvailable backups:")
            backup_dir = Path("chess_robot/backups")
            if backup_dir.exists():
                for backup in sorted(backup_dir.iterdir()):
                    if backup.is_dir():
                        print(f"  - {backup.name}")
        else:
            manager.restore_backup(args.name)
    elif args.command == 'clear':
        manager.clear_progress(keep_files=not args.delete_files)
    elif args.command == 'export':
        manager.export_metadata()


if __name__ == "__main__":
    main() 