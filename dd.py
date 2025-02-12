import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# time values
t_start, t_via, t_end = (0, 2, 6)

# start, via, and end
start_pt = np.array([5, 0])
via_pt   = np.array([5.9, 4.5])
end_pt   = np.array([3, 9])

# obstacle centers
obs_center1 = np.array([2.5, 3.0])
obs_center2 = np.array([8.5, 8.0])
obs_radius  = 3.0

# 1. segment 1: position at t_start equals start_pt.
# 2. segment 1: position at t_via equals via_pt.
# 3. segment 2: position at t_via equals via_pt.
# 4. segment 2: position at t_end equals end_pt.
# 5. segment 1: derivative at t_start equals 0.
# 6. continuity of derivative at t_via: derivative from seg1 equals derivative from seg2.
# 7. continuity of acceleration at t_via: second derivative from seg1 equals second derivative from seg2.
# 8. segment 2: derivative at t_end equals 0.
M = np.array([
    [t_start**3, t_start**2, t_start, 1, 0, 0, 0, 0],
    [t_via**3, t_via**2, t_via, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, t_via**3, t_via**2, t_via, 1],
    [0, 0, 0, 0, t_end**3, t_end**2, t_end, 1],
    [3*t_start**2, 2*t_start, 1, 0, 0, 0, 0, 0],
    [3*t_via**2, 2*t_via, 1, 0, -3*t_via**2, -2*t_via, -1, 0],
    [6*t_via, 2, 0, 0, -6*t_via, -2, 0, 0],
    [0, 0, 0, 0, 3*t_end**2, 2*t_end, 1, 0]
])


target_x = np.array([start_pt[0], via_pt[0], via_pt[0], end_pt[0], 0, 0, 0, 0])
target_y = np.array([start_pt[1], via_pt[1], via_pt[1], end_pt[1], 0, 0, 0, 0])
coeffs_x = np.linalg.solve(M, target_x)
coeffs_y = np.linalg.solve(M, target_y)

seg1_coeffs_x = coeffs_x[:4]
seg1_coeffs_y = coeffs_y[:4]
seg2_coeffs_x = coeffs_x[4:]
seg2_coeffs_y = coeffs_y[4:]

t_full = np.linspace(t_start, t_end, 100)
t_seg1 = t_full[t_full <= t_via]
t_seg2 = t_full[t_full > t_via]

x_seg1 = seg1_coeffs_x[0]*t_seg1**3 + seg1_coeffs_x[1]*t_seg1**2 + seg1_coeffs_x[2]*t_seg1 + seg1_coeffs_x[3]
y_seg1 = seg1_coeffs_y[0]*t_seg1**3 + seg1_coeffs_y[1]*t_seg1**2 + seg1_coeffs_y[2]*t_seg1 + seg1_coeffs_y[3]

x_seg2 = seg2_coeffs_x[0]*t_seg2**3 + seg2_coeffs_x[1]*t_seg2**2 + seg2_coeffs_x[2]*t_seg2 + seg2_coeffs_x[3]
y_seg2 = seg2_coeffs_y[0]*t_seg2**3 + seg2_coeffs_y[1]*t_seg2**2 + seg2_coeffs_y[2]*t_seg2 + seg2_coeffs_y[3]

traj_x = np.concatenate((x_seg1, x_seg2))
traj_y = np.concatenate((y_seg1, y_seg2))

# plot the trajectory and obstacles
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(traj_x, traj_y, 'b-', label='Trajectory')

# plot key points
ax.plot(start_pt[0], start_pt[1], 'go', markersize=8, label='Start')
ax.plot(via_pt[0], via_pt[1], 'o', color='orange', markersize=8, label='Via')
ax.plot(end_pt[0], end_pt[1], 'p', color='purple', markersize=10, label='End')

# plot obstacle circles
obs1_circle = plt.Circle(obs_center1, obs_radius, color='red', alpha=0.3)
obs2_circle = plt.Circle(obs_center2, obs_radius, color='red', alpha=0.3)
ax.add_artist(obs1_circle)
ax.add_artist(obs2_circle)

# Mark obstacle centers.
ax.plot(obs_center1[0], obs_center1[1], 'ro', markersize=8, label='Obstacle Center 1')
ax.plot(obs_center2[0], obs_center2[1], 'ro', markersize=8, label='Obstacle Center 2')

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('End-Effector Trajectory with Obstacles')
ax.legend(loc='best')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
plt.show()

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

ax1.plot(t_full, traj_x, 'b-', label='X Position')
ax1.set_ylabel('X Position')
ax1.set_title('X Position Over Time')
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.legend()

ax2.plot(t_full, traj_y, 'b-', label='Y Position')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Y Position')
ax2.set_title('Y Position Over Time')
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.legend()

plt.tight_layout()
plt.show()








