import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import sympy as sp
from sympy import Matrix, nsimplify

def dh_matrix(theta, d, alpha, a, symbolic=False):
    if symbolic:
        return sp.Matrix([
            [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
            [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
            [0,              sp.sin(alpha),                sp.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])
    else:
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])
    

def get_final_orr_pos(dh_table):
    num_joints = dh_table.shape[0]
    T_total = np.eye(4)
    
    for i in range(num_joints):
        theta, d, alpha, a = dh_table[i]
        T_link = dh_matrix(theta, d, alpha, a, symbolic = True)
        T_total = T_total @ T_link  
        
    print("Final End-Effector Transform (Symbolic): ")
    sp.pprint(sp.nsimplify(T_total.evalf(chop =1e-10, n=4)))


def get_joint_positions(dh_table):
    num_joints = dh_table.shape[0]
    T_total = np.eye(4)
    
    xs, ys, zs = [0], [0], [0]
    
    for i in range(num_joints):
        theta, d, alpha, a = dh_table[i]
        T_link = dh_matrix(theta, d, alpha, a, symbolic = False)
        
        T_total = T_total @ T_link  
        
        xs.append(T_total[0, 3])
        ys.append(T_total[1, 3])
        zs.append(T_total[2, 3])
        
    return xs, ys, zs

# sim setup 
# DH Table: [theta, d, alpha, a]
l1, l2, l3, l4 = 1.0, 1.0, 0.5, 0.5
#l1, l2, l3, l4 = sp.symbols('l1 l2 l3 l4')

base_dh = np.array([
    [-np.pi/2,  l1,     -np.pi/2,   0       ], # joint 1
    [-np.pi/2,  0,      0,          l2      ], # joint 2
    [np.pi/2,   0,      np.pi/2,    0       ], # joint 3
    [0,         l3,     -np.pi/2,   0       ], # joint 4
    [0,         0,      np.pi/2,    0       ], # joint 5
    [np.pi/2,   l4,     0,          0       ] # joint 6
])

joint_ranges = [
    (0, 2*np.pi),       # Joint 1
    (0, np.pi/2),     # Joint 2
    (-np.pi/4, np.pi/4), # Joint 3
    (0, np.pi),       # Joint 4
    (-np.pi/2, np.pi/2), # Joint 5
    (0, 2*np.pi)      # Joint 6
]

sequence = [
    (1, 0, np.pi/2, np.pi/4),      # Joint 2
    (0, 0, np.pi, 0),        # Joint 1
    (2, -0.5, 0.5, np.pi/4),       # Joint 3
    (4, 0, np.pi, np.pi/4),      # Joint 5
    (3, 0, np.pi*2, 0),      # Joint 4
    (5, 0, np.pi*2, 0)       # Joint 6
]

all_frames = []
current_offsets = np.zeros(6)

for joint_idx, start, middle, end in sequence:
    steps = np.linspace(start, middle, 20)
    for val in steps:
        current_offsets[joint_idx] = val
        all_frames.append(current_offsets.copy())
        
    steps_back = np.linspace(middle, end, 20)
    for val in steps_back:
        current_offsets[joint_idx] = val
        all_frames.append(current_offsets.copy())

# Setup Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_zlim([0, 3.0]) 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

line, = ax.plot([], [], [], 'o-', lw=3, color='blue')

current_offsets = np.zeros(6)

# --- Animation Loop ---
def update(frame_offsets):
    current_dh = base_dh.copy()
    current_dh[:, 0] += frame_offsets
    
    xs, ys, zs = get_joint_positions(current_dh)
    
    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    return line

print(f"Starting animation with {len(all_frames)} frames...")

# --- Annimate ---
# interval=50ms / frame
ani = FuncAnimation(fig, update, frames=all_frames, interval=50, blit=False)

plt.show()

# --- Print final end-effector position w/ variables ---
# get_final_orr_pos(base_dh)
