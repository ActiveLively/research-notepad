import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize, differential_evolution

deg_to_rad = np.pi / 180

# Defining limits as (min, max)
fanuc_limits = [
    (-170 * deg_to_rad, 170 * deg_to_rad), 
    (-75  * deg_to_rad, 75  * deg_to_rad),
    (-177 * deg_to_rad, 177 * deg_to_rad), 
    (-190 * deg_to_rad, 190 * deg_to_rad),
    (-100 * deg_to_rad, 100 * deg_to_rad),
    (-360 * deg_to_rad, 360 * deg_to_rad) 
]

def dh_matrix(theta, d, alpha, a):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def get_joint_positions(dh_table, joint_offsets):
    """Calculates the x, y, z coordinates of every joint for plotting."""
    T_total = np.eye(4)
    xs, ys, zs = [0], [0], [0]
    
    for i in range(len(dh_table)):
        theta = dh_table[i][0] + joint_offsets[i]
        d, alpha, a = dh_table[i][1], dh_table[i][2], dh_table[i][3]
        
        T_link = dh_matrix(theta, d, alpha, a)
        T_total = T_total @ T_link  
        
        xs.append(T_total[0, 3])
        ys.append(T_total[1, 3])
        zs.append(T_total[2, 3])
        
    return xs, ys, zs

def get_end_effector_matrix(joint_offsets):
    """Calculates the full 4x4 transformation matrix of the end-effector."""
    T_total = np.eye(4)
    for i in range(len(base_dh)):
        theta = base_dh[i][0] + joint_offsets[i]
        d, alpha, a = base_dh[i][1], base_dh[i][2], base_dh[i][3]
        T_total = T_total @ dh_matrix(theta, d, alpha, a)
    return T_total

# Robot Geometry
l1, l2, l3, l4 = 0.22, 0.24, 0.27, 0.07
base_dh = np.array([
    [0,  l1,  -np.pi/2,  0], 
    [0,  0,   0,         l2], 
    [0,  0,   np.pi/2,   0], 
    [0,  l3,  -np.pi/2,  0], 
    [0,  0,   np.pi/2,   0], 
    [0,  l4,  0,         0] 
])

def get_end_effector(joint_offsets):
    end_effector_matrix = get_end_effector_matrix(joint_offsets)
    return end_effector_matrix[:3, 3]

# target_pos = np.array([-0.3, 0.3, 0.55]) 
target_pos = np.array([-0.35, -0.35, 0.4]) 


target_rot = np.array([
    [ 1,  0,  0],
    [ 0, 1,  0],
    [ 0,  0, 1]
])

initial_guess = np.zeros(6)

def calc_error(q):
    T_ee = get_end_effector_matrix(q)
    
    # Extract current position 
    current_pos = T_ee[:3, 3]
    # Extract current rotation 
    current_rot = T_ee[:3, :3]
    
    pos_error = np.linalg.norm(current_pos - target_pos)
    rot_error = np.linalg.norm(current_rot - target_rot)
    
    # Combine the errors. 
    total_error = pos_error + (0.2 * rot_error) 
    
    return total_error

local_solution = minimize(calc_error, initial_guess, method='SLSQP', bounds=fanuc_limits)

global_solution = differential_evolution(calc_error, bounds=fanuc_limits, seed=42)

print(f"Target Position: {target_pos}")
print(f"Local Solver End Effector: {get_end_effector(local_solution.x).round(4)}")
print(f"Global Solver End Effector: {get_end_effector(global_solution.x).round(4)}")
print(f"Distance from target: Local: {calc_error(local_solution.x):.4f}, Global: {calc_error(global_solution.x):.4f}")

optimal_q = global_solution.x 

# 2. Generate animation frames by interpolating from the initial guess to the optimal solution
num_frames = 60
all_frames = []
for i in range(num_frames):
    fraction = i / (num_frames - 1)
    # Linearly interpolate between the start (all zeros) and the target angles
    frame_q = initial_guess + fraction * (optimal_q - initial_guess)
    all_frames.append(frame_q)

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Adjusted limits to better fit your smaller link lengths (0.22, 0.24, etc.)
ax.set_xlim([-0.8, 0.8])
ax.set_ylim([-0.8, 0.8])
ax.set_zlim([0, 1.0])

line, = ax.plot([], [], [], 'o-', lw=4, color='#2c3e50', markersize=8)

# 4. Plot the Target Position as a red star so we can visually verify it hits the mark
ax.scatter(*target_pos, color='red', s=200, marker='*', label='Target Position')
ax.legend()

def update(frame_offsets):
    xs, ys, zs = get_joint_positions(base_dh, frame_offsets)
    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    return line

ani = FuncAnimation(fig, update, frames=all_frames, interval=40, blit=False)
plt.title("FANUC CR-4iA Inverse Kinematics")
plt.show()