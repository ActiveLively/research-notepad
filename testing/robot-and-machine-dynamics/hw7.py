import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import CubicSpline

deg_to_rad = np.pi / 180

# Defining limits as (min, max)
# fanuc_limits = [
#     (-170 * deg_to_rad, 170 * deg_to_rad), 
#     (-75  * deg_to_rad, 75  * deg_to_rad),
#     (-177 * deg_to_rad, 177 * deg_to_rad), 
#     (-190 * deg_to_rad, 190 * deg_to_rad),
#     (-100 * deg_to_rad, 100 * deg_to_rad),
#     (-360 * deg_to_rad, 360 * deg_to_rad) 
# ]

fanuc_limits = [
    (-170 * deg_to_rad, 170 * deg_to_rad), 
    (-75  * deg_to_rad, 75  * deg_to_rad),
    (-177 * deg_to_rad, 177 * deg_to_rad), 
    (0, 0), 
    (0, 0),
    (0, 0)  
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

def get_jacobian_numeric(q, delta=1e-5):
    J = np.zeros((3, 6))
    ee_pos_base = get_end_effector(q)
    
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += delta
        ee_pos_plus = get_end_effector(q_plus)
        # Forward difference approximation
        J[:, i] = (ee_pos_plus - ee_pos_base) / delta
        
    return J

def calculate_static_torques(q):
    # 20 lbs = ~9.07 kg -> ~88.99 N
    F_load = np.array([0.0, 0.0, -88.99]) 
    
    J_v = get_jacobian_numeric(q)
    
    # Torque to counteract the load: tau_load = -J^T * F
    tau_load = -J_v.T @ F_load 
    
    # 2 lbs per link -> 0.907 kg -> 8.9 N
    m_kg = 2 * 0.453592
    m1 = m2 = m3 = m_kg
    
    l1 = 0.33
    l2 = 0.26
    l3 = 0.02
    g = 9.81
    
    C2 = np.cos(q[1])
    C23 = np.cos(q[1] + q[2])
    
    tau_g = np.zeros(6)
    
    # c1, c2, c3 equations from derivation
    tau_g[0] = 0.0  
    tau_g[1] = -0.5 * m2 * g * l2 * C2 - m3 * g * (l2 * C2 + 0.5 * l3 * C23) 
    tau_g[2] = -0.5 * m3 * g * l3 * C23 
    
    # 3. Total required joint torques
    tau_total = tau_g + tau_load
    
    tau_total[3] = 0.0
    tau_total[4] = 0.0
    tau_total[5] = 0.0
    
    return tau_total


def calc_error(q, target_pos=target_pos, target_rot=target_rot):
    T_ee = get_end_effector_matrix(q)
    
    current_pos = T_ee[:3, 3]
    # current_rot = T_ee[:3, :3]
    
    pose_error = np.linalg.norm(current_pos - target_pos)
    # rot_error = np.linalg.norm(current_rot - target_rot)
    # pose_error = pos_error + (0.2 * rot_error)
    
    tau = calculate_static_torques(q)
    torque_cost = np.sum(tau**2)
    
    penalty_weight = 1e6 
    total_cost = (penalty_weight * pose_error) + torque_cost
    
    return total_cost

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
q_waypoints = np.array(q_waypoints)    # shape (5, 6)

#  shortest angular path between waypoints
for i in range(1, len(q_waypoints)):
    diff = q_waypoints[i] - q_waypoints[i - 1]
    q_waypoints[i] -= np.round(diff / (2 * np.pi)) * 2 * np.pi

q_waypoints[-1] = q_waypoints[0]       # enforce identical endpoints for periodic spline

# Time allocated per segment (seconds) — equal durations
T_seg = 2.0
num_segments = num_points - 1

t_waypoints = np.arange(num_points) * T_seg
cs = CubicSpline(t_waypoints, q_waypoints, bc_type='periodic')

def sample_trajectory(points_per_seg=60):
    t_all   = np.linspace(0, num_segments * T_seg,
                          num_segments * points_per_seg, endpoint=False)
    q_all   = cs(t_all)
    qd_all  = cs(t_all, 1)
    qdd_all = cs(t_all, 2)
    ee_all  = np.array([get_end_effector(q) for q in q_all])
    return t_all, q_all, qd_all, qdd_all, ee_all
 
 
t_all, q_all, qd_all, qdd_all, ee_all = sample_trajectory(points_per_seg=60)
 
# ──────────────────────────────────────────────────────────────────────
# Print parametric equations for every segment & joint

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

print("Calculating torques for the generated trajectory...")
tau_all = np.array([calculate_static_torques(q) for q in q_all])

# Plot Joint Torques vs. Time
fig6, axes6 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
fig6.suptitle("Joint Torques vs. Time (Static Gravity + 20lb Load)", fontsize=14)

for j in range(6):
    ax = axes6[j // 2, j % 2]
    # Plotting torques in N-m
    ax.plot(t_all, tau_all[:, j], linewidth=2, color='tab:red')
    
    # Mark waypoint times
    for k in range(num_points):
        ax.axvline(k * T_seg, color='gray', linestyle='--', linewidth=0.5)
        
    ax.set_ylabel(f"τ{j+1} (N·m)")
    ax.grid(True, alpha=0.3)

axes6[2, 0].set_xlabel("Time (s)")
axes6[2, 1].set_xlabel("Time (s)")
fig6.tight_layout()
 
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