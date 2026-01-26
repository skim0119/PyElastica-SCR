from collections import defaultdict

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

import elastica as ea
import elastica_scr as es

final_time = 40.0

n_elem = 4  # Change based on requirements, but be careful
n_elem += n_elem % 2
half_n_elem = n_elem // 2

origin = np.zeros((3, 1))
angle_of_inclination = np.deg2rad(15.0)

# in-plane
horizontal_direction = np.array([0.0, 0.0, 1.0]).reshape(-1, 1)
vertical_direction = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)

# out-of-plane
normal = np.array([0.0, 1.0, 0.0])

total_length = 3.0
base_radius = 0.25
density = 5000
youngs_modulus = 1e4
poisson_ratio = 0.5
shear_modulus = youngs_modulus / (poisson_ratio + 1.0)

positions = np.empty((3, n_elem + 1))
dl = total_length / n_elem

# First half of positions stem from slope angle_of_inclination
first_half = np.arange(half_n_elem + 1.0).reshape(1, -1)
positions[..., : half_n_elem + 1] = origin + dl * first_half * (
    np.cos(angle_of_inclination) * horizontal_direction
    + np.sin(angle_of_inclination) * vertical_direction
)
positions[..., half_n_elem:] = positions[
    ..., half_n_elem : half_n_elem + 1
] + dl * first_half * (
    np.cos(angle_of_inclination) * horizontal_direction
    - np.sin(angle_of_inclination) * vertical_direction
)

butterfly_rod = es.StableCosseratRod.straight_rod(
    n_elem,
    start=origin.reshape(3),
    direction=np.array([0.0, 0.0, 1.0]),
    normal=normal,
    base_length=total_length,
    base_radius=base_radius,
    density=density,
    youngs_modulus=youngs_modulus,
    shear_modulus=shear_modulus,
    position=positions,
)


# Callback Setup
recorded_history: dict[str, list] = defaultdict(list)
recorded_history["time"].append(0.0)
recorded_history["position"].append(butterfly_rod.position_collection.copy())
recorded_history["te"].append(butterfly_rod.compute_translational_energy())
recorded_history["re"].append(butterfly_rod.compute_rotational_energy())
recorded_history["se"].append(butterfly_rod.compute_shear_energy())
recorded_history["be"].append(butterfly_rod.compute_bending_energy())


def callback(sim_time):
    recorded_history["time"].append(sim_time)
    # Collect x
    recorded_history["position"].append(butterfly_rod.position_collection.copy())
    # Collect energies as well
    recorded_history["te"].append(butterfly_rod.compute_translational_energy())
    recorded_history["re"].append(butterfly_rod.compute_rotational_energy())
    recorded_history["se"].append(butterfly_rod.compute_shear_energy())
    recorded_history["be"].append(butterfly_rod.compute_bending_energy())


# Finalize and Run
timestepper = es.SCRFirstOrderImplicit()

time = 0.0
dt = 0.01 * dl
total_steps = int(final_time / dt)
print("Total steps", total_steps)
dt = final_time / total_steps
print(f"{dt=}")
record_every = 100
for current_step in tqdm(range(total_steps)):
    time = timestepper.step_single_instance(butterfly_rod, time, dt)
    if current_step % record_every == 0:
        callback(time)

# Post-Processing

# Plot the histories
fig = plt.figure(figsize=(5, 4), frameon=True, dpi=150)
ax = fig.add_subplot(111)
positions_history = recorded_history["position"]
# record first position
first_position = positions_history.pop(0)
ax.plot(first_position[2, ...], first_position[0, ...], "r--", lw=2.0)
n_positions = len(positions_history)
for i, pos in enumerate(positions_history):
    alpha = np.exp(i / n_positions - 1)
    ax.plot(pos[2, ...], pos[0, ...], "b", lw=0.6, alpha=alpha)
# final position is also separate
last_position = positions_history.pop()
ax.plot(last_position[2, ...], last_position[0, ...], "k--", lw=2.0)
plt.savefig("position_overlay.png", dpi=300)
plt.close("all")

# Plot the energies
energy_fig = plt.figure(figsize=(5, 4), frameon=True, dpi=150)
energy_ax = energy_fig.add_subplot(111)
times = np.asarray(recorded_history["time"])
te = np.asarray(recorded_history["te"])
re = np.asarray(recorded_history["re"])
be = np.asarray(recorded_history["be"])
se = np.asarray(recorded_history["se"])

energy_ax.plot(times, te, c=to_rgb("xkcd:reddish"), lw=1.0, label="Translational")
energy_ax.plot(times, re, c=to_rgb("xkcd:bluish"), lw=1.0, label="Rotation")
energy_ax.plot(times, be, c=to_rgb("xkcd:burple"), lw=1.0, label="Bend")
energy_ax.plot(times, se, c=to_rgb("xkcd:goldenrod"), lw=1.0, label="Shear")
energy_ax.plot(times, te + re + be + se, c="k", lw=1.0, label="Total energy")
energy_ax.legend()
plt.savefig("energy.png", dpi=300)
plt.close("all")
