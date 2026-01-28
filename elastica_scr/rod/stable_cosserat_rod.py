__doc__ = """Rod Implementation"""
__all__ = ["StableCosseratRod", "StableCosseratRodWithBC"]
from typing import Any, Optional, Type

from numpy.typing import NDArray

import numpy as np
from elastica._calculus import difference_kernel_for_block_structure

from elastica.rod.cosserat_rod import CosseratRod

from .equations import (
    _update_geometry,
    _update_accelerations,
    _compute_internal_forces,
    _compute_internal_torques,
    _compute_bending_twist_strains,
    _compute_internal_torques_without_kappa_update,
)
from .._so3 import exp_so3, hat


class StableCosseratRod(CosseratRod):
    """ """

    REQUISITE_MODULES: list[Type] = []

    def compute_internal_forces_and_torques(self, time: np.float64) -> None:
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.

        Parameters
        ----------
        time: np.float64
            current time

        """
        _compute_internal_forces(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.dilatation,
            self.voronoi_dilatation,
            self.director_collection,
            self.sigma,
            self.rest_sigma,
            self.shear_matrix,
            self.internal_stress,
            self.internal_forces,
            self.ghost_elems_idx,
        )

        _compute_internal_torques(
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

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time: np.float64) -> None:
        """
        Updates the acceleration variables

        Parameters
        ----------
        time: np.float64
            current time

        """
        _update_accelerations(
            self.acceleration_collection,
            self.internal_forces,
            self.external_forces,
            self.mass,
            self.alpha_collection,
            self.inv_mass_second_moment_of_inertia,
            self.internal_torques,
            self.external_torques,
            self.dilatation,
        )

    def update_geometry(self, time: np.flaot64 | None = None):
        _update_geometry(
            self.position_collection,
            self.volume,
            self.lengths,
            self.tangents,
            self.radius,
            self.dilatation,
            self.rest_lengths,
            self.rest_voronoi_lengths,
            self.voronoi_dilatation,
        )

    def update_orientation(
        self, time: np.float64, dt: np.float64 | None = None
    ) -> list[float]:
        """
        orientation is treated as quasi-static. dt should not matter
        """
        # Previous residual calculation
        new_dmds, T = _compute_internal_torques(
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
        reg = 1e-3
        y = -(
            (T + self.external_torques).T.flatten()
        )  # T shape is (3, N_v), and we want to iterate column-wise

        m_kron, _, _, _ = np.linalg.lstsq(
            A.T.dot(A) + reg * np.identity(3 * (self.n_elems - 1)), A.T.dot(y)
        )
        m = m_kron.reshape(self.n_elems - 1, 3).T

        k = m.copy()
        k[0, :] /= self.bend_matrix[0, 0, :]
        k[1, :] /= self.bend_matrix[1, 1, :]
        k[2, :] /= self.bend_matrix[2, 2, :]

        # Find orientation
        sor = 0.7
        old_kappa = np.empty_like(self.kappa)
        old_director = self.director_collection.copy()
        for _ in range(50):
            _compute_bending_twist_strains(
                old_director,
                self.rest_voronoi_lengths,
                old_kappa,
            )

            axes = difference_kernel_for_block_structure(
                0.5 * (old_kappa + k), self.ghost_voronoi_idx
            )
            for nidx in range(self.n_elems):
                old_director[:, :, nidx] = (
                    exp_so3(-sor * axes[:, nidx]) @ old_director[:, :, nidx]
                )

        kappa_residual = np.linalg.norm(old_kappa - k, axis=0).mean()
        director_difference = np.linalg.norm(
            old_director - self.director_collection, ord="fro", axis=(0, 1)
        ).mean()

        # print(
        #    f"kappa residual: {kappa_residual:.7f}, dir diff: {director_difference:.7f}"
        # )

        # Update parameters
        self.kappa[:] = k
        self.director_collection[:] = old_director
        self.internal_couple[:] = m

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
            k,  # self.kappa,
            self.voronoi_dilatation,
            self.mass_second_moment_of_inertia,
            self.omega_collection,
            self.internal_stress,
            m,  # self.internal_couple,
            self.dilatation,
            self.dilatation_rate,
            self.internal_torques,
            self.ghost_voronoi_idx,
        )

        after_value = np.linalg.norm(self.internal_torques, axis=0).mean()
        # print(f"{prev_value:.04f} -> {after_value:.04f}")

    def update_position_and_velocity(self, time, dt):
        delh = np.zeros((self.n_nodes, self.n_elems))
        np.fill_diagonal(delh, 1)
        np.fill_diagonal(delh[1:], -1)
        delh_kron = np.zeros((self.n_nodes * 3, self.n_elems * 3))
        delh_kron[0::3, 0::3] = delh
        delh_kron[1::3, 1::3] = delh
        delh_kron[2::3, 2::3] = delh
        diff = np.zeros((self.n_elems, self.n_nodes))
        np.fill_diagonal(diff, -1)
        np.fill_diagonal(diff[:, 1:], 1)
        diff_kron = np.zeros((self.n_elems * 3, self.n_nodes * 3))
        diff_kron[0::3, 0::3] = diff
        diff_kron[1::3, 1::3] = diff
        diff_kron[2::3, 2::3] = diff
        QTS_kron = np.zeros((self.n_elems * 3, self.n_elems * 3))
        eQ_kron = np.zeros((self.n_elems * 3, self.n_elems * 3))
        z_kron = np.zeros((self.n_elems * 3))
        z_kron[2::3] = 1.0
        for elem_idx in range(self.n_elems):
            sidx = 3 * elem_idx
            QTS_kron[sidx : sidx + 3, sidx : sidx + 3] = (
                self.director_collection[:, :, elem_idx].T
                @ self.shear_matrix[:, :, elem_idx]
                / self.dilatation[elem_idx]
            )
            eQ_kron[sidx : sidx + 3, sidx : sidx + 3] = (
                self.director_collection[:, :, elem_idx] * self.dilatation[elem_idx]
            )
        reg = 1e-3

        dmdtv = (self.velocity_collection * (self.mass / dt)[None, :]).T.flatten()
        dmdt2r = (self.position_collection * (self.mass / (dt**2))[None, :]).T.flatten()
        A = delh_kron @ QTS_kron @ eQ_kron @ diff_kron - np.diag(
            np.repeat(self.mass / (dt**2), 3)
        )
        y = (
            delh_kron @ QTS_kron @ z_kron
            + self.external_forces.T.ravel()
            - dmdtv
            - dmdt2r
        )

        r_kron, _, _, _ = np.linalg.lstsq(
            A.T.dot(A) + reg * np.identity(3 * self.n_nodes), A.T.dot(y)
        )
        position_new = r_kron.reshape(self.n_nodes, 3).T

        self.velocity_collection[:] = (position_new - self.position_collection) / dt
        self.position_collection[:] = position_new
