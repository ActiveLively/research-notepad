import roboticstoolbox as rtb
import numpy as np
import matplotlib.pyplot as plt

# 1. Define Robot Parameters
l1, l2, l3, l4 = 1.0, 1.0, 0.5, 0.5

# 2. Create the Robot using Standard DH
# We map your MATLAB parameters to the Revolute class:
# MATLAB: [a, alpha, d, theta_offset]
# Python: Revolute(a, alpha, d, offset)

robot = rtb.DHRobot([
    # Joint 1: a=0, alpha=-pi/2, d=l1, offset=-pi/2
    rtb.Revolute(a=0, alpha=-np.pi/2, d=l1, offset=-np.pi/2, qlim=[-np.pi, np.pi]),
    
    # Joint 2: a=l2, alpha=0, d=0, offset=-pi/2
    rtb.Revolute(a=l2, alpha=0, d=0, offset=-np.pi/2, qlim=[-np.pi/4, np.pi/4]),
    
    # Joint 3: a=0, alpha=pi/2, d=0, offset=pi/2
    rtb.Revolute(a=0, alpha=np.pi/2, d=0, offset=np.pi/2, qlim=[-np.pi/2, np.pi/2]),
    
    # Joint 4: a=pi/2, alpha=-pi/2, d=l3, offset=0
    rtb.Revolute(a=np.pi/2, alpha=-np.pi/2, d=l3, offset=0, qlim=[-np.pi/2, np.pi/2]),
    
    # Joint 5: a=0, alpha=pi/2, d=0, offset=0
    rtb.Revolute(a=0, alpha=np.pi/2, d=0, offset=0, qlim=[-np.pi/2, np.pi/2]),
    
    # Joint 6: a=0, alpha=0, d=l4, offset=pi/2
    rtb.Revolute(a=0, alpha=0, d=l4, offset=np.pi/2, qlim=[-np.pi/2, np.pi/2])
], name="MyRobot")

print(robot)

# 3. Setup Visualization
# We use the 'pyplot' backend (matplotlib) which is simple and reliable.
# For a fancier browser-based 3D view, change backend to 'swift'.
env = robot.plot(robot.qz, backend='pyplot', block=False)

print("Starting Sequential Animation...")

# 4. Sequential Animation Loop
while True:
    # Start with all joints at zero
    q_current = np.zeros(6)
    
    # Loop through each joint index (0 to 5)
    for i in range(6):
        # Get limits for this specific joint
        min_lim, max_lim = robot.links[i].qlim
        
        print(f"Moving Joint {i+1} from {min_lim:.2f} to {max_lim:.2f}")
        
        # Create a smooth trajectory from min to max
        # jtraj generates a path with smooth acceleration/deceleration
        traj = rtb.tools.trajectory.jtraj(min_lim, max_lim, 30).q
        
        # 1. Sweep Forward
        for val in traj:
            q_current[i] = val # Update only the current joint
            robot.plot(q_current, backend='pyplot', dt=0.01)
            
        # 2. Sweep Backward (using the reverse of the trajectory)
        for val in reversed(traj):
            q_current[i] = val
            robot.plot(q_current, backend='pyplot', dt=0.01)
            
        # Optional: Reset to 0 before moving the next joint
        # q_current[i] = 0 
        # robot.plot(q_current, backend='pyplot')
        
    print("Sequence complete! Restarting...")