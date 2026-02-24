__doc__ = """Rod Implementation"""
__all__ = ["StableCosseratRod", "StableCosseratRodWithBC"]
from typing import Any, Optional, Type

from numpy.typing import NDArray

import numpy as np
import numba
from .stable_cosserat_rod import StableCosseratRodWithBC
from elastica._calculus import (
    difference_kernel_for_block_structure,
    _trapezoidal_for_block_structure,
)
from elastica._linalg import _batch_cross
from elastica._rotations import _rotate

from .equations import (
    _update_geometry,
    _update_accelerations,
    _compute_internal_forces,
    _compute_internal_torques,
    _compute_bending_twist_strains,
    _compute_internal_torques_without_kappa_update,
)
from .._so3 import exp_so3, hat, _compute_relative_rotation


class CROEF(StableCosseratRodWithBC):
    """One-end-fixed Stable Cosserat Rod"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix(fixed_position_indices=(0,), fixed_director_indices=(0,))
        self.num_bc = 2

    def update_orientation(self, time: np.float64, dt: np.float64) -> list[float]:
        """
        orientation is treated as quasi-static. dt should not matter.

        fit:
        0 = m_s + kappa x m + Qr_s/e x Ssigma + C
        sigma = Qr_s - z

        given: r_s, S, C, B
        find: kappa, m, and Q
        relate: m=Bkappa, Q_s = m

        """
        return
        # Previous residual calculation
        new_dmds, T = _compute_internal_torques_without_kappa_update(
            # new_dmds, T = _compute_internal_torques(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.bend_matrix,
            self.rest_kappa,
            self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )
        prev_value = np.linalg.norm(self.internal_torques, axis=0).mean()

        # from .lie_integrator import adjoint_midpoint_euler, inplace_euler_top_flow

        delh = np.zeros((self.n_elems, self.n_elems - 1))
        np.fill_diagonal(delh, 1)
        np.fill_diagonal(delh[1:], -1)
        delh_kron = np.zeros((self.n_elems * 3, (self.n_elems - 1) * 3))
        delh_kron[0::3, 0::3] = delh
        delh_kron[1::3, 1::3] = delh
        delh_kron[2::3, 2::3] = delh
        Ah = np.zeros((self.n_elems, self.n_elems - 1))
        np.fill_diagonal(Ah, 0.5)
        np.fill_diagonal(Ah[1:], 0.5)
        Ah_kron = np.zeros((self.n_elems * 3, (self.n_elems - 1) * 3))
        Ah_kron[0::3, 0::3] = Ah
        Ah_kron[1::3, 1::3] = Ah
        Ah_kron[2::3, 2::3] = Ah

        e3 = 1.0 / self.voronoi_dilatation**3
        e3_kron = np.repeat(e3, 3)
        de3 = self.rest_voronoi_lengths * e3
        de3_kron = np.repeat(de3, 3)

        kappa_kron = np.zeros(((self.n_elems - 1) * 3, (self.n_elems - 1) * 3))
        for voro_idx in range(self.n_elems - 1):
            sidx = 3 * voro_idx
            kappa_x = hat(self.kappa[:, voro_idx])
            kappa_kron[sidx : sidx + 3, sidx : sidx + 3] = kappa_x

        A = delh_kron * e3_kron[None, :] + (Ah_kron * de3_kron[None, :]) @ kappa_kron
        reg = 0.0  # 1e-6
        y = -(
            (T + self.external_torques).T.flatten()
        )  # T shape is (3, N_v), and we want to iterate column-wise

        m_kron, couple_residual, _, _ = np.linalg.lstsq(
            A.T.dot(A) + reg * np.identity(3 * (self.n_elems - 1)), A.T.dot(y)
        )
        updated_internal_couple = m_kron.reshape(self.n_elems - 1, 3).T

        target_kappa = np.empty_like(self.kappa)
        target_kappa[:] = updated_internal_couple
        target_kappa[0, :] /= self.bend_matrix[0, 0, :]
        target_kappa[1, :] /= self.bend_matrix[1, 1, :]
        target_kappa[2, :] /= self.bend_matrix[2, 2, :]

        # Re-evaluate
        new_dmds, T = _compute_internal_torques_without_kappa_update(
            # new_dmds, T = _compute_internal_torques(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.bend_matrix,
            self.rest_kappa,
            target_kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            updated_internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )
        mid_value = np.linalg.norm(self.internal_torques, axis=0).mean()

        inv_moi = np.diagonal(self.inv_mass_second_moment_of_inertia).T
        rotational_damping_constant = 0.0  # 2.6e3
        _rotational_damping_coefficient = np.exp(
            -rotational_damping_constant * inv_moi * dt
        )
        dissipation_ratio = np.power(_rotational_damping_coefficient, self.dilatation)

        # Find orientation (Integrate kappa from base to tip)
        updated_kappa = target_kappa.copy()
        updated_director = _integrate_director_from_kappa(
            self.director_collection[:, :, 0],
            updated_kappa,
            self.rest_voronoi_lengths,
        )
        _compute_bending_twist_strains(
            updated_director,
            self.rest_voronoi_lengths,
            updated_kappa,
        )
        # Curvature Kappa is known -- updated_kappa is found from the previous routine, directly finding solution to the ODE with implicit method.
        # Write algorithm: Given that the simulation has fixed director at the base, rest of the director should be
        # obtained by just integrating kappa.

        updated_omega = _compute_relative_rotation(
            self.director_collection, updated_director
        )
        updated_omega /= dt

        kappa_residual = np.linalg.norm(target_kappa - updated_kappa, axis=0).mean()
        director_difference = np.linalg.norm(
            updated_director - self.director_collection, ord="fro", axis=(0, 1)
        ).mean()
        omega_difference = np.linalg.norm(
            updated_omega - self.omega_collection, axis=0
        ).mean()

        voronoi_length = self.rest_voronoi_lengths * self.voronoi_dilatation
        print(f"shear mag: {np.linalg.norm(T, axis=0)}")
        print(
            f"(prev)curvature mag: {np.linalg.norm(self.kappa, axis=0) * voronoi_length * 180 / np.pi} deg"
        )
        print(
            f"curvature mag: {np.linalg.norm(updated_kappa, axis=0) * voronoi_length * 180 / np.pi} deg"
        )
        print(
            f"(target)curvature mag: {np.linalg.norm(target_kappa, axis=0) * voronoi_length * 180 / np.pi} deg"
        )
        print(
            f"internal couple mag: {np.linalg.norm(updated_internal_couple, axis=0)}, residual:  {couple_residual}"
        )
        print(
            f"omega mag: {np.linalg.norm(updated_omega, axis=0) * dt * 180 / np.pi} deg"
        )
        print(f"kappa residual: {kappa_residual}")
        for k in range(updated_director.shape[-1]):
            print(updated_director[2, :, k])

        # Update parameters
        self.kappa[:] = updated_kappa
        self.internal_couple[:] = updated_internal_couple
        self.director_collection[:] = updated_director
        self.alpha_collection[:] = (updated_omega - self.omega_collection) / dt
        self.omega_collection[:] = updated_omega

        # Re-evaluation of internal_torques --> supposedly zero
        dmds, T = _compute_internal_torques_without_kappa_update(
            self.position_collection,
            self.velocity_collection,
            self.tangents,
            self.lengths,
            self.rest_lengths,
            self.director_collection,
            self.rest_voronoi_lengths,
            self.bend_matrix,
            self.rest_kappa,
            updated_kappa,  # self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            updated_internal_couple,  # self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )

        after_value = np.linalg.norm(self.internal_torques, axis=0).mean()
        print(
            f"{prev_value:.04f} -> {mid_value:.04f} -> {after_value:.04f}, shear term: {np.linalg.norm(T, axis=0).mean():.04f}\n"
        )


@numba.njit(cache=True)  # type: ignore
def _integrate_director_from_kappa(
    base_director: NDArray[np.float64],
    kappa: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Integrate director frames from base to tip using known voronoi curvature/twist.
    """
    n_elems = kappa.shape[1] + 1
    updated_director = np.empty((3, 3, n_elems))
    updated_director[:, :, 0] = base_director

    for k in range(n_elems - 1):
        rotation = exp_so3(kappa[:, k] * rest_voronoi_lengths[k])
        updated_director[:, :, k + 1] = rotation @ updated_director[:, :, k]

    return updated_director
