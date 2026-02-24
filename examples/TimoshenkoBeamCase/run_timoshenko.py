import numpy as np
from collections import defaultdict
from tqdm import tqdm
import elastica as ea
import elastica_scr as es
from elastica.version import VERSION

from timoshenko_postprocessing import plot_timoshenko


class TimoshenkoBeamSimulator(
    ea.BaseSystemCollection, ea.Constraints, ea.Forcing, ea.CallBacks, ea.Damping
):
    pass


timoshenko_sim = TimoshenkoBeamSimulator()

# setting up test params
simulation_time = 500  # 5000.0  # (sec)

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
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 99
shear_modulus = E / (poisson_ratio + 1.0)

shearable_rod = es.CROEF.straight_rod(
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
timoshenko_sim.append(shearable_rod)


dl = base_length / n_elem
dt = 0.07 * dl
print(f"dt: {dt:.04f}")
timoshenko_sim.dampen(shearable_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)


# # One end of the rod is now fixed in place
# timoshenko_sim.constrain(shearable_rod).using(
#     ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
# )

# Forces added to the rod
end_force = np.array([-15.0, 0.0, 0.0])
timoshenko_sim.add_forcing_to(shearable_rod).using(
    ea.EndpointForces, 0.0 * end_force, end_force, ramp_up_time=simulation_time / 2.0
)


# Add call backs
class Callback(ea.CallBackBaseClass):
    """
    Tracks rod position every fixed amount of simulation time.
    """

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
timoshenko_sim.collect_diagnostics(shearable_rod).using(
    Callback, sample_every_seconds=0.5, callback_params=recorded_history
)


timoshenko_sim.finalize()
# timestepper = ea.PositionVerlet()
timestepper = es.SCROEFImplicit()

total_steps = int(simulation_time / dt)
print("Total steps", total_steps)

time = 0.0
for i in tqdm(range(total_steps)):
    time = timestepper.step(timoshenko_sim, time, dt)

plot_timoshenko(shearable_rod, end_force, False, True, recorded_history)
