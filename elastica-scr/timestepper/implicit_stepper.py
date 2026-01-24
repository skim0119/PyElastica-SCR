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

from ..systems.protocol import ImplicitsystemProtocol


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
        # TODO : Implement
        # simulation_time = np.float64(time)
        # simulation_dt = np.float64(dt)

        # for kin_prefactor, kin_step, dyn_step in self.steps_and_prefactors[:-1]:
        #     for system in SystemCollection.final_systems():
        #         kin_step(system, simulation_time, simulation_dt)

        #     simulation_time += kin_prefactor(simulation_dt)

        #     # Constrain only values
        #     SystemCollection.constrain_values(simulation_time)

        #     # We need internal forces and torques because they are used by interaction module.
        #     for system in SystemCollection.final_systems():
        #         system.compute_internal_forces_and_torques(simulation_time)

        #     # Add external forces, controls etc.
        #     SystemCollection.synchronize(simulation_time)

        #     for system in SystemCollection.final_systems():
        #         dyn_step(system, simulation_time, simulation_dt)

        #     # Constrain only rates
        #     SystemCollection.constrain_rates(simulation_time)

        # # Peel the last kinematic step and prefactor alone
        # last_kin_prefactor = self.steps_and_prefactors[-1][0]
        # last_kin_step = self.steps_and_prefactors[-1][1]

        # for system in SystemCollection.final_systems():
        #     last_kin_step(system, simulation_time, simulation_dt)
        # simulation_time += last_kin_prefactor(simulation_dt)
        # SystemCollection.constrain_values(simulation_time)

        # # Call back function, will call the user defined call back functions and store data
        # SystemCollection.apply_callbacks(
        #     simulation_time, round(simulation_time / simulation_dt)
        # )

        # # Zero out the external forces and torques
        # for system in SystemCollection.final_systems():
        #     system.zeroed_out_external_forces_and_torques(simulation_time)

        return time + dt

    def step_single_instance(
        self,
        System: ImplicitSystemType,
        time: np.float64,
        dt: np.float64,
    ) -> np.float64:
        """
        (The function is used for single system instance, mainly for testing purposes.)
        """

        # Quasi-static orientation solve
        residual = system.update_orientation(
            director,
            kappa,
            sigma,
            tangent,
            voronoi_length,
            voronoi_dilatation,
            dilatation,
            external_couple,
            shear_matrix,
            bend_matrix,
        )
        self.info["orientation_residual"] = residual

        # Implicit position solve
        residual = system.update_position()
        self.info["position_residual"] = residual

        # Velocity update
        system.update_velocity()

        return time + dt
