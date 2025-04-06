import h5py
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Read HDF5 Data Sets
# ----------------------
def read_velocity_data(file_path):
    with h5py.File(file_path, 'r') as f:
        u = np.array(f["Velocity"]["X-comp"])
        v = np.array(f["Velocity"]["Y-comp"])
    return u, v

# ----------------------
# Calculate Magnitudes
# ----------------------
def compute_velocity_magnitude(u, v):
    return np.sqrt(u**2 + v**2)

def compute_vorticity_magnitude(u, v):
    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=0)
    return np.abs(dv_dx - du_dy)

u, v = read_velocity_data("data/isabel_2d.h5")

velocity_magnitude = compute_velocity_magnitude(u, v)
vorticity_magnitude = compute_vorticity_magnitude(u, v)

plt.figure(figsize=(8, 6))
plt.imshow(velocity_magnitude, origin='upper', cmap='plasma')
plt.axis('off')
plt.savefig("img/isabel_velocity_magnitude.png", bbox_inches='tight', pad_inches=0)
plt.title("Isabel Velocity Magnitude")
plt.colorbar(label="Velocity Magnitude")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(vorticity_magnitude, origin='upper', cmap='viridis')
plt.axis('off')
plt.savefig("img/isabel_vorticity_magnitude.png", bbox_inches='tight', pad_inches=0)
plt.title("Isabel Vorticity Magnitude")
plt.colorbar(label="Vorticity Magnitude")
plt.tight_layout()
plt.show()
