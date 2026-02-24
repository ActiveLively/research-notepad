import time
import math
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# Configuration
ROBOT_IP = "192.168.56.101"  # Update to your UR20's actual IP
SPEED = 1.0                   # 10 cm/s - Keep it slow for safety
ACCEL = 1.0                  # Slow acceleration
BLEND = 0.01                 # 1 cm blend radius for smooth corners

# Infinity Sign Parameters
AMPLITUDE_X = 0.15           # Controls total width (0.15m * 2 = 30cm wide)
AMPLITUDE_Y = 0.15           # Controls total height (approx 15cm high)
NUM_POINTS = 50              # Number of waypoints to approximate the curve

try:
    # Initialize interfaces
    receiver = RTDEReceiveInterface(ROBOT_IP)
    control = RTDEControlInterface(ROBOT_IP)

    # 1. Get current TCP Pose: [x, y, z, rx, ry, rz]
    current_pose = receiver.getActualTCPPose()
    print(f"Starting Position: {current_pose}")

    # Extract current coordinates as our center point
    cx, cy, cz = current_pose[0], current_pose[1], current_pose[2]
    rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]

    path = []

    # 2. Calculate the waypoints for the infinity sign
    print("Calculating trajectory...")
    for i in range(NUM_POINTS + 1):
        # t goes from 0 to 2*PI
        t = (i / NUM_POINTS) * 2 * math.pi
        
        # Lemniscate of Gerono parametric equations
        dx = AMPLITUDE_X * math.sin(t)
        dy = AMPLITUDE_Y * math.sin(t) * math.cos(t)
        
        nx = cx + dx
        ny = cy + dy
        
        # The blend radius for the very last point MUST be 0.0 so the robot stops
        current_blend = 0.0 if i == NUM_POINTS else BLEND
        
        # Format for moveL path: [x, y, z, rx, ry, rz, speed, accel, blend]
        waypoint = [nx, ny, cz, rx, ry, rz, SPEED, ACCEL, current_blend]
        path.append(waypoint)

    # 3. Execute the full blended path
    print(f"Moving along infinity path using {len(path)} waypoints...")
    # Passing a list of lists triggers the blended path motion in ur_rtde
    control.moveL(path)

    print("Movement complete.")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Always clean up the connection
    if 'control' in locals():
        control.stopScript()