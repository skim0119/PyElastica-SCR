__doc__ = """Symplectic time steppers and concepts for integrating the kinematic and dynamic equations of rod-like objects.  """

from typing import TYPE_CHECKING, Any, Callable

from itertools import zip_longest

from elastica.typing import (
    SystemCollectionType,
    SystemType,
    StepType,
    SteppersOperatorsType,
)

import numpy as np

from ..systems.protocol import ImplicitSystemProtocol


class SCRFirstOrderImplicit:
    """
    Implicit stepper for differential-algebraic equations.
    Only handles translational implicit for Cosserat rod states.
    Rotational components are handled internally. Check stable Cosserat Rod assumptions.
    """

    def __init__(self):
        # placeholder for stepping information
        self.info: dict[str, Any] = {}

    def step(
        self,
        SystemCollection: SystemCollectionType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        Function for doing symplectic stepper over the user defined rods (system).

        Returns
        -------
        time: float
            The time after the integration step.

        """
        simulation_time = np.float64(time)
        simulation_dt = np.float64(dt)

        # Update geometry
        for system in SystemCollection.final_systems():
            system.update_geometry()

        SystemCollection.synchronize(simulation_time)

        # Quasi-static orientation solve
        for system in SystemCollection.final_systems():
            system.update_orientation(time)
        SystemCollection.constrain_rates(simulation_time)

        # Implicit position and velocity
        for system in SystemCollection.final_systems():
            system.update_position_and_velocity(time, dt)
        SystemCollection.constrain_values(simulation_time)

        # Call back function, will call the user defined call back functions and store data
        SystemCollection.apply_callbacks(
            simulation_time, round(simulation_time / simulation_dt)
        )

        # Zero out the external forces and torques
        for system in SystemCollection.final_systems():
            system.zeroed_out_external_forces_and_torques(simulation_time)

        return time + dt

    def step_single_instance(
        self,
        system: ImplicitSystemProtocol,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        (The function is used for single system instance, mainly for testing purposes.)
        """

        # Update geometry
        system.update_geometry()

        # Quasi-static orientation solve
        orientation_residual = system.update_orientation(time)
        self.info["orientation_residual"] = np.array(orientation_residual)

        # Implicit position and velocity
        position_residual = system.update_position_and_velocity(time, dt)
        self.info["position_residual"] = position_residual

        return time + dt
