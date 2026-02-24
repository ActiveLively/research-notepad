import time
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Configuration
ROBOT_IP = "192.168.56.101"  # Update to your UR20's actual IP
SPEED = 0.1               # 10 cm/s - Keep it slow for safety
ACCEL = 0.1                # Slow acceleration
Z_OFFSET = 0.25             # 0.1 meters (approx 4 inches)

try:
    # Initialize interfaces
    # rtde_receive handles reading data; rtde_control handles moving
    receiver = RTDEReceiveInterface(ROBOT_IP)
    control = RTDEControlInterface(ROBOT_IP)

    # 1. Get current TCP Pose: [x, y, z, rx, ry, rz]
    current_pose = receiver.getActualTCPPose()
    print(f"Current Position: {current_pose}")

    # 2. Calculate new pose
    new_pose = list(current_pose)
    new_pose[2] += Z_OFFSET  # Index 2 is the Z-axis

    print(f"Moving to: {new_pose}")

    # 3. Execute MoveL (Linear motion)
    # moveL(pose, speed, acceleration, asynchronous)
    control.moveL(new_pose, SPEED, ACCEL)

    print("Movement complete.")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Always clean up the connection
    if 'control' in locals():
        control.stopScript()