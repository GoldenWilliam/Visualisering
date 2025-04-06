import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import os

# ----------------------
# Read HDF5 Data Sets
# ----------------------
def read_velocity_data(file_path):
    with h5py.File(file_path, 'r') as f:
        u = np.array(f["Velocity"]["X-comp"])
        v = np.array(f["Velocity"]["Y-comp"])
    return u, v

# ----------------------
# Interpolation Function
# ----------------------
def interpolate_velocity(u, v, x, y):
    ny, nx = u.shape
    x_vals = np.linspace(0, nx - 1, nx)
    y_vals = np.linspace(0, ny - 1, ny)
    u_interp = RegularGridInterpolator((y_vals, x_vals), u, bounds_error=False, fill_value=np.nan)
    v_interp = RegularGridInterpolator((y_vals, x_vals), v, bounds_error=False, fill_value=np.nan)
    return lambda pos: np.array([u_interp((pos[1], pos[0])), v_interp((pos[1], pos[0]))])

# ----------------------
# LIC-compatible Euler Integrator
# ----------------------
def lic_euler_integrator(start_pos, h, steps, velocity_field, nx, ny):
    path = [start_pos]
    pos = np.array(start_pos, dtype=float)
    for _ in range(steps):
        v = velocity_field(pos)
        if np.any(np.isnan(v)) or np.linalg.norm(v) == 0 or np.any(pos < 0) or pos[0] >= nx or pos[1] >= ny:
            break
        v_unit = v / np.linalg.norm(v)
        pos = pos + h * v_unit
        path.append(pos.copy())
    return np.array(path)

# ----------------------
# LIC-compatible RK4 Integrator
# ----------------------
def lic_rk4_integrator(start_pos, h, steps, velocity_field, nx, ny):
    path = [start_pos]
    pos = np.array(start_pos, dtype=float)

    for _ in range(steps):
        def unit_v(p):
            v = velocity_field(p)
            if np.any(np.isnan(v)) or np.linalg.norm(v) == 0:
                return np.array([np.nan, np.nan])
            return v / np.linalg.norm(v)

        k1 = unit_v(pos)
        k2 = unit_v(pos + 0.5 * h * k1)
        k3 = unit_v(pos + 0.5 * h * k2)
        k4 = unit_v(pos + h * k3)

        if np.any(np.isnan([k1, k2, k3, k4])):
            break

        delta = (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pos = pos + delta

        if np.any(pos < 0) or pos[0] >= nx or pos[1] >= ny:
            break

        path.append(pos.copy())
    return np.array(path)


# ----------------------
# LIC Visualization (High Detail)
# ----------------------
def plot_lic_texture(velocity_func, nx, ny, integrator, steps=300, h=0.2, apply_smoothing=False, colormap='bone', gamma=1.2, upscale_factor=2, save_path=None):
    def bidirectional_path(start, h, steps, velocity_field, nx, ny):
        forward = integrator(start, h, steps // 2, velocity_field, nx, ny)
        backward = integrator(start, -h, steps // 2, velocity_field, nx, ny)
        backward = backward[::-1][:-1] if len(backward) > 1 else np.empty((0, 2))
        return np.concatenate((backward, forward), axis=0)

    start_time = time.time()
    up_ny, up_nx = ny * upscale_factor, nx * upscale_factor
    highres_noise = np.random.rand(up_ny, up_nx)
    noise = gaussian_filter(highres_noise, sigma=0.3)

    lic_result = np.full((up_ny, up_nx), np.nan)
    num_samples = np.zeros((up_ny, up_nx))

    for j in tqdm(range(up_ny), desc="LIC rows", unit="row", dynamic_ncols=True, colour='cyan'):
        for i in range(up_nx):
            start = (i / upscale_factor, j / upscale_factor)
            path = bidirectional_path(start, h, steps, velocity_func, nx, ny)
            vals = []
            from scipy.ndimage import map_coordinates

            for pt in path:
                if not np.any(np.isnan(pt)):
                    if 0 <= pt[0] < up_nx and 0 <= pt[1] < up_ny:
                        val = map_coordinates(noise, [[pt[1]], [pt[0]]], order=1, mode='nearest')[0]
                        vals.append(val)
            if vals:
                lic_result[j, i] = np.mean(vals)
                num_samples[j, i] = len(vals)

    if apply_smoothing:
        lic_result = gaussian_filter(lic_result, sigma=0.3)

    lic_min = np.nanmin(lic_result)
    lic_max = np.nanmax(lic_result)
    lic_normalized = (lic_result - lic_min) / (lic_max - lic_min)
    lic_gamma = np.power(lic_normalized, gamma)

    plt.figure(figsize=(10, 10))
    plt.imshow(lic_gamma, origin='upper', cmap=colormap, extent=[0, nx, ny, 0])
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

    elapsed = time.time() - start_time
    print(f"\033[95mLIC completed in {elapsed:.2f} seconds.\033[0m")
    avg_time_per_row = elapsed / ny
    estimated_total = avg_time_per_row * ny
    print(f"\033[94mAverage time per row: {avg_time_per_row:.2f} seconds\033[0m")
    print(f"\033[96mEstimated total time: {estimated_total:.2f} seconds\033[0m")

# ----------------------
# Main
# ----------------------
u, v = read_velocity_data("data/metsim1_2d.h5")
ny, nx = u.shape
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
velocity_func = interpolate_velocity(u, v, x, y)

plot_lic_texture(
    velocity_func, nx, ny, lic_rk4_integrator,
    steps=50, h=0.2,
    apply_smoothing=False,
    colormap='bone',
    gamma=1.2,
    save_path="img/metsim_lic_rk4_50.png"
)
