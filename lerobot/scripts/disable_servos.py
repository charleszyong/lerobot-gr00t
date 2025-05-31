#!/usr/bin/env python3
"""
Emergency script to disable SO100 robot servos.
Use this if servos remain enabled after a crash or failed shutdown.
"""

import sys
import time
import argparse

def disable_so100_servos(port="/dev/ttyACM1"):
    """Directly disable SO100 servos without going through the full robot stack."""
    try:
        import scservo_sdk as scs
    except ImportError:
        print("Error: scservo_sdk not found. Please install with: pip install scservo-sdk")
        return False
    
    print(f"Attempting to disable servos on port {port}...")
    
    # Initialize port
    port_handler = scs.PortHandler(port)
    packet_handler = scs.PacketHandler(0)  # Protocol version 0
    
    try:
        # Open port
        if not port_handler.openPort():
            print(f"Failed to open port {port}")
            print("Make sure the robot is connected and you have the correct permissions")
            print("Try: sudo chmod 666 /dev/ttyACM*")
            return False
        
        port_handler.setBaudRate(1_000_000)
        port_handler.setPacketTimeoutMillis(1000)
        
        # SO100 typically has 6 motors with IDs 1-6
        motor_ids = [1, 2, 3, 4, 5, 6]
        torque_enable_addr = 40  # Address for Torque_Enable in STS3215
        
        print("Disabling torque on each motor...")
        success_count = 0
        
        for motor_id in motor_ids:
            try:
                # Write 0 to disable torque
                comm = packet_handler.write1ByteTxRx(port_handler, motor_id, torque_enable_addr, 0)
                if comm == scs.COMM_SUCCESS:
                    print(f"  Motor {motor_id}: Torque disabled ✓")
                    success_count += 1
                else:
                    print(f"  Motor {motor_id}: Failed - {packet_handler.getTxRxResult(comm)}")
            except Exception as e:
                print(f"  Motor {motor_id}: Error - {e}")
        
        print(f"\nDisabled torque on {success_count}/{len(motor_ids)} motors")
        
        # Close port
        port_handler.closePort()
        
        if success_count > 0:
            print("\nServos should now be free to move. You can safely power off the robot.")
            return True
        else:
            print("\nWARNING: Could not disable any servos! Please power off the robot manually.")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        if port_handler:
            port_handler.closePort()
        return False


def main():
    parser = argparse.ArgumentParser(description="Emergency servo disable for SO100 robot")
    parser.add_argument("--port", default="/dev/ttyACM1", help="Serial port (default: /dev/ttyACM1)")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports")
    args = parser.parse_args()
    
    if args.list_ports:
        import serial.tools.list_ports
        print("Available serial ports:")
        for port in serial.tools.list_ports.comports():
            print(f"  {port.device} - {port.description}")
        return
    
    print("SO100 Emergency Servo Disable")
    print("=" * 40)
    print("This will attempt to disable all servo motors.")
    print("Use this if servos are still powered after a crash.\n")
    
    success = disable_so100_servos(args.port)
    
    if not success:
        print("\nIf this script fails, you can:")
        print("1. Try a different port with --port")
        print("2. List available ports with --list-ports")
        print("3. Power off the robot at the power supply")
        sys.exit(1)


if __name__ == "__main__":
    main() 