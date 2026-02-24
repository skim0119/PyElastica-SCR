import numpy as np
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt

import elastica as ea
import elastica_scr as es


class AxialStretchingSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks, ea.Damping
):
    pass


axial_sim = AxialStretchingSimulator()

# Match the Timoshenko setup, except for force direction.
simulation_time = 20.0
n_elem = 100
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius**2
density = 5000
nu = 0.1 / 7 / density / base_area
E = 1e6
poisson_ratio = 99
shear_modulus = E / (poisson_ratio + 1.0)

stretchable_rod = es.CROEF.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)
axial_sim.append(stretchable_rod)


dl = base_length / n_elem
dt = 0.07 * dl
print(f"dt: {dt:.04f}")
axial_sim.dampen(stretchable_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

# Axial stretching force: along rod axis (+z), unlike Timoshenko's transverse load.
end_force = np.array([0.0, 0.0, 15.0])
axial_sim.add_forcing_to(stretchable_rod).using(
    ea.EndpointForces, 0.0 * end_force, end_force, ramp_up_time=simulation_time / 2.0
)


class Callback(ea.CallBackBaseClass):
    """Tracks rod states every fixed amount of simulation time."""

    def __init__(self, sample_every_seconds: float, callback_params: dict):
        super().__init__()
        self.sample_every_seconds = sample_every_seconds
        self.callback_params = callback_params
        self.next_sample_time = 0.0

    def make_callback(self, system, time, current_step: int):
        if time + 1e-12 >= self.next_sample_time:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["director"].append(system.director_collection.copy())
            self.next_sample_time += self.sample_every_seconds
            return


recorded_history = defaultdict(list)
axial_sim.collect_diagnostics(stretchable_rod).using(
    Callback, sample_every_seconds=1.0, callback_params=recorded_history
)


axial_sim.finalize()
timestepper = es.SCROEFImplicit()

total_steps = int(simulation_time / dt)
print("Total steps", total_steps)

time = 0.0
for _ in tqdm(range(total_steps)):
    time = timestepper.step(axial_sim, time, dt)

# Plot tip axial position (z) over time.
times = np.asarray(recorded_history["time"], dtype=float)
positions = recorded_history["position"]
tip_z = np.asarray([position[2, -1] for position in positions], dtype=float)

expected_tip_disp = end_force[2] * base_length / base_area / E
expected_tip_disp_improved = end_force[2] * base_length / (base_area * E - end_force[2])

fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
ax = fig.add_subplot(111)
ax.plot(times, tip_z, lw=2.0, label="tip z")
ax.hlines(
    base_length + expected_tip_disp,
    0.0,
    simulation_time,
    colors="k",
    linestyles="dashdot",
    lw=1.0,
    label="linear estimate",
)
ax.hlines(
    base_length + expected_tip_disp_improved,
    0.0,
    simulation_time,
    colors="k",
    linestyles="dashed",
    lw=2.0,
    label="improved estimate",
)
ax.set_xlabel("time [s]")
ax.set_ylabel("tip z position [m]")
ax.set_title("Axial Stretching (SCR/CROEF)")
ax.legend()
plt.show()
