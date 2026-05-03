import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = np.array([
    0, 20, 40, 60, 80,          # approach straight
    100, 120, 140,              # S-bend start
    160, 180,                   # S-bend exit
    200, 210, 215, 210, 200,    # hairpin loop
    180, 160, 140, 120, 100     # exit
])

y = np.array([
     0,  5, 15,  5, -10,
    -20, -5, 15,
     25, 10,
     -5, 15, 40, 65, 85,
     95, 90, 70, 50, 30
])

track_width = 15
half_width = track_width/2

t = np.zeros(len(x))

for i in range(1, len(x)):
    cur = np.sqrt((x[i] - x[i-1])**2 + (y[i]-y[i-1])**2)
    t[i] = t[i-1] + cur

cs_x = CubicSpline(t, x)
cs_y = CubicSpline(t, y)

nodes = np.linspace(t[0], t[-1], 100)

x_s = cs_x(nodes)
y_s = cs_y(nodes)

dx = np.gradient(x_s)
dy = np.gradient(y_s)

mag = np.sqrt(dx**2 + dy**2)
tx = dx/mag
ty = dy/mag

nx = -ty
ny = tx

d = np.zeros_like(x_s)

alpha = 0.2

for it in range(100):

    x_r = x_s + d * nx
    y_r = y_s + d * ny

    for i in range(len(x_s)):
        left = (i-1) % len(x_s)
        right = (i+1) % len(x_s)

        left_vector = np.array([x_r[left] - x_r[i], y_r[left] - y_r[i]])
        right_vector = np.array([x_r[right] - x_r[i], y_r[right] - y_r[i]])
        left_vector /= np.linalg.norm(left_vector)
        right_vector /= np.linalg.norm(right_vector)

        bisector = left_vector + right_vector
        norm_b = np.linalg.norm(bisector)
        if norm_b < 1e-6:
            continue
        bisector /= norm_b

        normal = np.array([nx[i], ny[i]])
        dp = np.dot(bisector, normal)
        d[i] += alpha * dp
        d[i] = np.clip(d[i], -half_width, half_width)
        d[i] = 0.9 * d[i] + 0.1 * (d[i] + alpha * dp)


inner_x = x_s - nx * half_width
inner_y = y_s - ny * half_width

outer_x = x_s + nx * half_width
outer_y = y_s + ny * half_width

plt.figure(figsize=(10, 8))

# track area
plt.fill_between(outer_x, inner_y, outer_y, alpha=0.2, color='gray')

# boundaries
plt.plot(inner_x, inner_y, 'k', linewidth=2)
plt.plot(outer_x, outer_y, 'k', linewidth=2)

# centerline
plt.plot(x_s, y_s, '--', label='centerline')

# racing line
plt.plot(x_r, y_r, 'r', linewidth=2, label='racing line')

plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
