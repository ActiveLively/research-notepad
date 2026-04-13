import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

class DraggableBSpline:
    def __init__(self):
        self.control_points = np.array([
            [0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [3.0, 2.0], [4.0, 0.0], [5.0, 1.0], [6.0, -2.0]
        ])
        
        self.p = 3  
        
        # Setup figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title('Controllable Cubic B-Spline')
        self.ax.set_xlim(-1, 8)
        self.ax.set_ylim(-4, 5)
        self.ax.grid(True)
        
        self.poly_line, = self.ax.plot([], [], 'ro--', alpha=0.5, markersize=8, label='Control Points')
        self.curve_line, = self.ax.plot([], [], 'b-', linewidth=2.5, label='B-Spline Curve')
        self.ax.legend()
        
        self.dragging_point_idx = None
        self.epsilon = 0.3  
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.update_plot()

    def get_bspline_curve(self):
        num_points = len(self.control_points)
        n = num_points - 1
        k = self.p + 1
        
        knots = np.concatenate(([0] * k, np.linspace(0, 1, n - k + 3)[1:-1], [1] * k))
        
        bspline = BSpline(knots, self.control_points, self.p)
        u = np.linspace(0, 1, 100)
        return bspline(u)

    def update_plot(self):
        self.poly_line.set_data(self.control_points[:, 0], self.control_points[:, 1])
        
        # Update curve
        curve_pts = self.get_bspline_curve()
        self.curve_line.set_data(curve_pts[:, 0], curve_pts[:, 1])
        
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        
        x, y = event.xdata, event.ydata
        distances = np.sqrt(np.sum((self.control_points - np.array([x, y]))**2, axis=1))
        min_idx = np.argmin(distances)
        
        if distances[min_idx] < self.epsilon:
            self.dragging_point_idx = min_idx

    def on_release(self, event):
        self.dragging_point_idx = None

    def on_motion(self, event):
        if self.dragging_point_idx is None or event.inaxes != self.ax: return
        
        self.control_points[self.dragging_point_idx] = [event.xdata, event.ydata]
        self.update_plot()

if __name__ == '__main__':
    app = DraggableBSpline()
    plt.show()