import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


# Parametric Cubic Curve
t = np.linspace(0, 1, 100)

a0_0, a1_0, a2_0, a3_0 = 1, 12, -30, 20
a0_1, a1_1, a2_1, a3_1 = 1, 12, -24, 20
a0_2, a1_2, a2_2, a3_2 = 1, -12, 20, -30

q_a0 = a0_0 + a1_0*t + a2_0*(t**2) + a3_0*(t**3)
q_a1 = a0_1 + a1_1*t + a2_1*(t**2) + a3_1*(t**3)
q_a2 = a0_2 + a1_2*t + a2_2*(t**2) + a3_2*(t**3)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t, q_a0, label='v1', linewidth=2)
plt.plot(t, q_a1, label='v2, a2 -> -24', linestyle='--', linewidth=2)
plt.plot(t, q_a2, label='v3, a1 -> -12, a2 -> 20, a3 -> -30', linestyle=':', linewidth=2)
plt.title('Parametric Cubic Curve Comparison')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)




# ----------------------------
# Cubic B-Spline Demonstration

control_points_base = np.array([[0, 0], [1, 3], [2, -1], [3, 2], [4, 0], [5, 1], [6, -2]])

control_points_changed = np.copy(control_points_base)
control_points_changed[2, 1] = 4 # Change the third point to [2, 4]

# Cubic B-spline
p = 3 # Degree
k = p + 1 # Degree + 1
num_points = len(control_points_base)
n = num_points - 1 # Number of control points - 1
num_knots = n + k + 1 
num_precision = n - k + 2


knots = np.concatenate(([0] * k, np.linspace(0, 1, n - k + 3)[1:-1], [1] * k))

# Generate B-splines
bspline_base = BSpline(knots, control_points_base, p)
bspline_changed = BSpline(knots, control_points_changed, p)

u = np.linspace(0, 1, 100)
curve_base = bspline_base(u)
curve_changed = bspline_changed(u)

plt.subplot(1, 2, 2)
# Plot base curve and points
plt.plot(curve_base[:, 0], curve_base[:, 1], label='Base Curve', linewidth=2)
plt.plot(control_points_base[:, 0], control_points_base[:, 1], 'o-', alpha=0.4, label='Base Control Polygon')

# Plot changed curve and points
plt.plot(curve_changed[:, 0], curve_changed[:, 1], linestyle='--', label='Changed Curve', linewidth=2)
plt.plot(control_points_changed[:, 0], control_points_changed[:, 1], 'x--', alpha=0.6, label='Changed Polygon')

plt.title('Cubic B-Spline\n(Local Control)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()