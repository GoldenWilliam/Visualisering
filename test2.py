import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import RegularGridInterpolator
from matplotlib.collections import LineCollection
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
# Create Mesh Grid
# ----------------------
def create_mesh(u):
    ny, nx = u.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y

# ----------------------
# Calculate Magnitudes
# ----------------------
def compute_velocity_magnitude(u, v):
    return np.sqrt(u ** 2 + v ** 2)

def compute_vorticity_magnitude(u, v):
    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=0)
    return np.abs(dv_dx - du_dy)

# ----------------------
# Interpolation Function
# ----------------------
def interpolate_velocity(u, v, x, y):
    u_interp = RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=np.nan)
    v_interp = RegularGridInterpolator((y, x), v, bounds_error=False, fill_value=np.nan)
    return lambda pos: np.array([u_interp((pos[1], pos[0])), v_interp((pos[1], pos[0]))])

# ----------------------
# Streamline Integrators
# ----------------------
def euler_integrator(start_pos, h, steps, velocity_field, nx, ny):
    path = [start_pos]
    pos = np.array(start_pos, dtype=float)
    for _ in range(steps):
        v = velocity_field(pos)
        if np.any(np.isnan(v)) or np.any(pos < 0) or pos[0] >= nx or pos[1] >= ny:
            break
        pos = pos + h * np.array([v[0], v[1]])
        path.append(pos.copy())
    return np.array(path)

def rk4_integrator(start_pos, h, steps, velocity_field, nx, ny):
    path = [start_pos]
    pos = np.array(start_pos, dtype=float)
    for _ in range(steps):
        k1 = h * velocity_field(pos)
        k2 = h * velocity_field(pos + 0.5 * k1)
        k3 = h * velocity_field(pos + 0.5 * k2)
        k4 = h * velocity_field(pos + k3)
        delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if np.any(np.isnan(delta)) or np.any(pos < 0) or pos[0] >= nx or pos[1] >= ny:
            break
        pos = pos + delta
        path.append(pos.copy())
    return np.array(path)

# ----------------------
# Seeding Strategies
# ----------------------
def uniform_seeds(nx, ny, count):
    xs = np.linspace(0, nx - 1, int(np.sqrt(count)))
    ys = np.linspace(0, ny - 1, int(np.sqrt(count)))
    return [(x, y) for x in xs for y in ys]

def random_seeds(nx, ny, count):
    np.random.seed(42)
    xs = np.random.uniform(0, nx - 1, count)
    ys = np.random.uniform(0, ny - 1, count)
    return list(zip(xs, ys))

def density_based_seeds(u, v, count):
    magnitude = compute_velocity_magnitude(u, v)
    flat_indices = np.argsort(magnitude.ravel())[::-1][:count]
    ny, nx = u.shape
    return [(idx % nx, idx // nx) for idx in flat_indices]

def flow_feature_seeds(u, v, count):
    vorticity = compute_vorticity_magnitude(u, v)
    vorticity[np.isnan(vorticity)] = 0
    threshold_vort = 0.5
    valid_mask = vorticity > threshold_vort
    valid_indices = np.argwhere(valid_mask)
    if len(valid_indices) == 0:
        return []
    n_clusters = min(count, len(valid_indices))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(valid_indices)
    centers = kmeans.cluster_centers_
    return [(float(x), float(y)) for y, x in centers]

# ----------------------
# Plotting Function
# ----------------------
def plot_field_lines(X, Y, u, v, seeds, velocity_func, nx, ny, integrator, steps=500, activate_streamplot=False, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    if activate_streamplot:
        speed = compute_velocity_magnitude(u, v)
        fig.patch.set_facecolor('white')
        strm = ax.streamplot(
            X, Y, u, v,
            color=speed,
            norm=plt.Normalize(vmin=global_min, vmax=global_max),
            linewidth=1.0,
            cmap='coolwarm',
            density=2.0,
            arrowsize=0.5
        )
        ax.set_xlim(0, nx - 1)
        ax.set_ylim(ny - 1, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()
        return

    for seed in seeds:
        path = integrator(seed, 0.5, steps, velocity_func, nx, ny)
        if len(path) > 1:
            magnitudes = [np.linalg.norm(velocity_func(pt)) if not np.any(np.isnan(velocity_func(pt))) else 0 for pt in path]
            points = path.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='coolwarm', norm=plt.Normalize(vmin=global_min, vmax=global_max))
            lc.set_array(np.array(magnitudes[:-1]))
            lc.set_linewidth(1.0)
            ax.add_collection(lc)

    ax.set_xlim(0, nx - 1)
    ax.set_ylim(ny - 1, 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

# ----------------------
# Main: Metsim Dataset
# ----------------------
global_min = None
global_max = None

os.makedirs("img", exist_ok=True)

u, v = read_velocity_data("data/metsim1_2d.h5")
x, y, X, Y = create_mesh(u)
nx, ny = len(x), len(y)

velocity_func = interpolate_velocity(u, v, x, y)
velocity_magnitude = compute_velocity_magnitude(u, v)
global_min = np.nanmin(velocity_magnitude)
global_max = np.nanmax(velocity_magnitude)

seeding_strategy = "uniform"
count = 49
if seeding_strategy == "uniform":
    seeds = uniform_seeds(nx, ny, count)
elif seeding_strategy == "random":
    seeds = random_seeds(nx, ny, count)
elif seeding_strategy == "density":
    seeds = density_based_seeds(u, v, count)
elif seeding_strategy == "flow_feature":
    seeds = flow_feature_seeds(u, v, count)
else:
    raise ValueError("Unknown seeding strategy")

plot_field_lines(
    X, Y, u, v, seeds, velocity_func, nx, ny, euler_integrator,
    steps=1000,
    save_path=f"img/metsim_fieldlines_euler_{seeding_strategy}_.png"
)

plot_field_lines(
    X, Y, u, v, seeds, velocity_func, nx, ny, rk4_integrator,
    steps=1000,
    save_path=f"img/metsim_fieldlines_rk4_{seeding_strategy}_.png"
)

# plot_field_lines(
#     X, Y, u, v, seeds, velocity_func, nx, ny, rk4_integrator,
#     activate_streamplot=True,
#     save_path="img/metsim_fieldlines_streamplot.png"
# )
