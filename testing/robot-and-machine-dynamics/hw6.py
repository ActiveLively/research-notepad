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

def calc_error(q, target_pos=target_pos, target_rot=target_rot):
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

def solve_ik(target_pos, target_rot, seed=None):
    result = differential_evolution(
        calc_error, bounds=fanuc_limits,
        args=(target_pos, target_rot), seed=seed,
        maxiter=1000, tol=1e-8
    )
    ee = get_end_effector(result.x)
    err = np.linalg.norm(ee - target_pos)
    print(f"  Target: {target_pos}  |  Achieved: {np.round(ee, 4)}  |  Error: {err:.6f}")
    return result.x

target_points = np.array([
    [ 0.20,  0.20,  0.45],
    [-0.25,  0.15,  0.50],
    [-0.30, -0.20,  0.35],
    [ 0.05, -0.05,  0.40],
    [ 0.20,  0.20,  0.45],
])

num_points = len(target_points)
print("=" * 60)
print("Solving Inverse Kinematics for each waypoint …")
print("=" * 60)
q_waypoints = []                      
for idx, wp in enumerate(target_points):
    print(f"\nWaypoint {idx + 1}:")
    q = solve_ik(wp, target_rot, seed=42 + idx)
    q_waypoints.append(q)
q_waypoints = np.array(q_waypoints)    # shape (4, 6)

#  shortest angular path between waypoints
for i in range(1, len(q_waypoints)):
    diff = q_waypoints[i] - q_waypoints[i - 1]
    q_waypoints[i] -= np.round(diff / (2 * np.pi)) * 2 * np.pi

#Cubic Polynomial Trajectory
def cubic_coefficients(q_start, q_end, T):
    # return coefficients for cubic polynomial
    dq = q_end - q_start
    a0 = q_start
    a1 = 0.0
    a2 =  3.0 * dq / T**2
    a3 = -2.0 * dq / T**3
    return np.array([a0, a1, a2, a3])

def eval_cubic(coeffs, t):
    """Evaluate θ(t) = a0 + a1·t + a2·t² + a3·t³."""
    a0, a1, a2, a3 = coeffs
    return a0 + a1*t + a2*t**2 + a3*t**3
 
def eval_cubic_vel(coeffs, t):
    """Evaluate θ'(t) = a1 + 2·a2·t + 3·a3·t²."""
    _, a1, a2, a3 = coeffs
    return a1 + 2*a2*t + 3*a3*t**2
 
def eval_cubic_acc(coeffs, t):
    """Evaluate θ''(t) = 2·a2 + 6·a3·t."""
    _, _, a2, a3 = coeffs
    return 2*a2 + 6*a3*t

# Time allocated per segment (seconds) — equal durations
T_seg = 2.0
num_segments = num_points - 1
 
# Tabulate coefficients for every (segment, joint) pair
coeffs_table = []
for seg in range(num_segments):
    seg_coeffs = []
    for j in range(6):
        c = cubic_coefficients(q_waypoints[seg, j],
                               q_waypoints[seg + 1, j], T_seg)
        seg_coeffs.append(c)
    coeffs_table.append(seg_coeffs)

def sample_trajectory(points_per_seg=60):
    t_all, q_all, qd_all, qdd_all, ee_all = [], [], [], [], []
 
    for seg in range(num_segments):
        t_local = np.linspace(0, T_seg, points_per_seg, endpoint=(seg == num_segments - 1))
        for t in t_local:
            q   = np.array([eval_cubic(coeffs_table[seg][j], t) for j in range(6)])  # joints
            qd  = np.array([eval_cubic_vel(coeffs_table[seg][j], t) for j in range(6)]) # velocities
            qdd = np.array([eval_cubic_acc(coeffs_table[seg][j], t) for j in range(6)]) # accelerations
            t_all.append(seg * T_seg + t)
            q_all.append(q)
            qd_all.append(qd)
            qdd_all.append(qdd)
            ee_all.append(get_end_effector(q))
 
    return (np.array(t_all), np.array(q_all),
            np.array(qd_all), np.array(qdd_all), np.array(ee_all))
 
 
t_all, q_all, qd_all, qdd_all, ee_all = sample_trajectory(points_per_seg=60)
 
# ──────────────────────────────────────────────────────────────────────
# Print parametric equations for every segment & joint
print("\n" + "=" * 60)
print("Cubic Polynomial Parametric Equations  (t in [0, {:.1f}] s)".format(T_seg))
print("=" * 60)
for seg in range(num_segments):
    print(f"\n--- Segment {seg+1}: Waypoint {seg+1} → Waypoint {seg+2} ---")
    for j in range(6):
        a0, a1, a2, a3 = coeffs_table[seg][j]
        print(f"  θ{j+1}(t) = {a0:+.6f} {a1:+.6f}·t {a2:+.6f}·t² {a3:+.6f}·t³")


# Plot Joint Angles vs. Time
fig1, axes1 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
fig1.suptitle("Joint Angles vs. Time (Cubic Polynomial Trajectories)", fontsize=14)
for j in range(6):
    ax = axes1[j // 2, j % 2]
    ax.plot(t_all, np.degrees(q_all[:, j]), linewidth=2)
    # Mark waypoint times
    for k in range(num_points):
        ax.axvline(k * T_seg, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel(f"θ{j+1} (deg)")
    ax.grid(True, alpha=0.3)
axes1[2, 0].set_xlabel("Time (s)")
axes1[2, 1].set_xlabel("Time (s)")
fig1.tight_layout()
 
# Plot Joint Velocities vs. Time
fig2, axes2 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
fig2.suptitle("Joint Velocities vs. Time", fontsize=14)
for j in range(6):
    ax = axes2[j // 2, j % 2]
    ax.plot(t_all, np.degrees(qd_all[:, j]), linewidth=2, color='tab:orange')
    for k in range(num_points):
        ax.axvline(k * T_seg, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel(f"ω{j+1} (deg/s)")
    ax.grid(True, alpha=0.3)
axes2[2, 0].set_xlabel("Time (s)")
axes2[2, 1].set_xlabel("Time (s)")
fig2.tight_layout()
 
# Plot Joint Accelerations vs. Time
fig3, axes3 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
fig3.suptitle("Joint Accelerations vs. Time", fontsize=14)
for j in range(6):
    ax = axes3[j // 2, j % 2]
    ax.plot(t_all, np.degrees(qdd_all[:, j]), linewidth=2, color='tab:green')
    for k in range(num_points):
        ax.axvline(k * T_seg, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel(f"α{j+1} (deg/s²)")
    ax.grid(True, alpha=0.3)
axes3[2, 0].set_xlabel("Time (s)")
axes3[2, 1].set_xlabel("Time (s)")
fig3.tight_layout()
 

# Plot End-Effector Path
fig4 = plt.figure(figsize=(8, 8))
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot(ee_all[:, 0], ee_all[:, 1], ee_all[:, 2],
         'b-', linewidth=2, label='End-Effector Path')
for idx, wp in enumerate(target_points):
    ax4.scatter(*wp, color='red', s=200, marker='*', zorder=5)
    ax4.text(wp[0], wp[1], wp[2] + 0.02, f"P{idx+1}", fontsize=10,
             fontweight='bold', color='red')
ax4.set_xlabel("X (m)")
ax4.set_ylabel("Y (m)")
ax4.set_zlabel("Z (m)")
ax4.set_title("End-Effector Cartesian Path")
ax4.legend()
fig4.tight_layout()
 
# 
# Video generation

fig5 = plt.figure(figsize=(9, 9))
ax5 = fig5.add_subplot(111, projection='3d')
ax5.set_xlim([-0.6, 0.6])
ax5.set_ylim([-0.6, 0.6])
ax5.set_zlim([0, 0.8])
ax5.set_xlabel("X (m)")
ax5.set_ylabel("Y (m)")
ax5.set_zlabel("Z (m)")
ax5.set_title("FANUC CR-4iA Path Animation")
 
# Draw waypoints
for idx, wp in enumerate(target_points):
    ax5.scatter(*wp, color='red', s=200, marker='*', zorder=5)
    ax5.text(wp[0], wp[1], wp[2] + 0.02, f"P{idx+1}", fontsize=10,
             fontweight='bold', color='red')
 
# Trace of end-effector
trace_line, = ax5.plot([], [], [], 'b--', linewidth=1, alpha=0.5)
robot_line, = ax5.plot([], [], [], 'o-', lw=4, color='#2c3e50', markersize=8)
 
trace_x, trace_y, trace_z = [], [], []
 
def animate(i):
    q = q_all[i]
    xs, ys, zs = get_joint_positions(base_dh, q)
    robot_line.set_data(xs, ys)
    robot_line.set_3d_properties(zs)
 
    trace_x.append(xs[-1])
    trace_y.append(ys[-1])
    trace_z.append(zs[-1])
    trace_line.set_data(trace_x, trace_y)
    trace_line.set_3d_properties(trace_z)
    return robot_line, trace_line
 
ani = FuncAnimation(fig5, animate, frames=len(q_all), interval=30, blit=False)
 
plt.show()