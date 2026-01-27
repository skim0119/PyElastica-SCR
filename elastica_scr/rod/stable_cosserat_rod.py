__doc__ = """Rod Implementation"""
from typing import Any, Optional, Type

from numpy.typing import NDArray

import numpy as np
import functools
import numba
from elastica.rod import RodBase
from elastica.systems.protocol import SystemProtocol
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
    _batch_matvec,
)
from elastica._rotations import _inv_rotate
from elastica._calculus import (
    quadrature_kernel_for_block_structure,
    difference_kernel_for_block_structure,
    _difference,
    _average,
)

from elastica.rod.cosserat_rod import CosseratRod
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


# Below is the numba-implementation of Cosserat Rod equations. They don't need to be visible by users.


@numba.njit(cache=True)  # type: ignore
def _update_geometry(
    position_collection: NDArray[np.float64],
    volume: NDArray[np.float64],
    lengths: NDArray[np.float64],
    tangents: NDArray[np.float64],
    radius: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
) -> None:
    """
    Update <length, tangents, and radius> given <position and volume>.
    Update <dilatation and voronoi_dilatation>
    """
    # Compute eq (3.3) from 2018 RSOS paper

    # Note : we can use the two-point difference kernel, but it needs unnecessary padding
    # and hence will always be slower
    position_diff = _difference(position_collection)
    # FIXME: Here 1E-14 is added to fix ghost lengths, which is 0, and causes division by zero error!
    lengths[:] = _batch_norm(position_diff) + 1e-14
    # _reset_scalar_ghost(lengths, ghost_elems_idx, 1.0)

    for k in range(lengths.shape[0]):
        tangents[0, k] = position_diff[0, k] / lengths[k]
        tangents[1, k] = position_diff[1, k] / lengths[k]
        tangents[2, k] = position_diff[2, k] / lengths[k]
        # resize based on volume conservation
        radius[k] = np.sqrt(volume[k] / lengths[k] / np.pi)

    # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
    # Put in initialization
    for k in range(lengths.shape[0]):
        dilatation[k] = lengths[k] / rest_lengths[k]

    # Compute eq (3.4) from 2018 RSOS paper
    # Note : we can use trapezoidal kernel, but it has padding and will be slower
    voronoi_lengths = _average(lengths)

    # Compute eq (3.4) from 2018 RSOS paper
    for k in range(voronoi_lengths.shape[0]):
        voronoi_dilatation[k] = voronoi_lengths[k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)  # type: ignore
def _compute_bending_twist_strains(
    director_collection: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    kappa: NDArray[np.float64],
) -> None:
    """
    Update <curvature/twist (kappa)> given <director and rest_voronoi_length>.
    """
    temp = _inv_rotate(director_collection)
    blocksize = rest_voronoi_lengths.shape[0]
    for k in range(blocksize):
        kappa[0, k] = temp[0, k] / rest_voronoi_lengths[k]
        kappa[1, k] = temp[1, k] / rest_voronoi_lengths[k]
        kappa[2, k] = temp[2, k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)  # type: ignore
def _compute_internal_bending_twist_stresses_from_model(
    director_collection: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    internal_couple: NDArray[np.float64],
    bend_matrix: NDArray[np.float64],
    kappa: NDArray[np.float64],
    rest_kappa: NDArray[np.float64],
) -> None:
    """
    Upate <internal couple> given <curvature(kappa) and bend_matrix>.

    Linear force functional
    Operates on
    B : (3,3,n) tensor and curvature kappa (3,n)
    """
    _compute_bending_twist_strains(
        director_collection, rest_voronoi_lengths, kappa
    )  # concept : needs to compute kappa

    blocksize = kappa.shape[1]
    temp = np.empty((3, blocksize))
    for i in range(3):
        for k in range(blocksize):
            temp[i, k] = kappa[i, k] - rest_kappa[i, k]

    internal_couple[:] = _batch_matvec(bend_matrix, temp)


@numba.njit(cache=True)  # type: ignore
def _compute_internal_forces(
    position_collection: NDArray[np.float64],
    volume: NDArray[np.float64],
    lengths: NDArray[np.float64],
    tangents: NDArray[np.float64],
    radius: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    sigma: NDArray[np.float64],
    rest_sigma: NDArray[np.float64],
    shear_matrix: NDArray[np.float64],
    internal_stress: NDArray[np.float64],
    internal_forces: NDArray[np.float64],
    ghost_elems_idx: NDArray[np.float64],
) -> None:
    """
    Update <internal force> given <director, internal_stress and velocity>.
    Update <shear/stretch(sigma)> given <dilatation, director, and tangent>.
    Update <internal stress> given <shear matrix, sigma, and rest_sigma>.

    Linear force functional
    Operates on
    S : (3,3,n) tensor and sigma (3,n)
    """

    # Compute n_l and cache it using internal_stress
    # Be careful about usage though
    # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
    z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
    sigma[:] = dilatation * _batch_matvec(director_collection, tangents) - z_vector
    internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)

    # Signifies Q^T n_L / e
    # Not using batch matvec as I don't want to take directors.T here

    blocksize = internal_stress.shape[1]
    cosserat_internal_stress = np.zeros((3, blocksize))

    for i in range(3):
        for j in range(3):
            for k in range(blocksize):
                cosserat_internal_stress[i, k] += (
                    director_collection[j, i, k] * internal_stress[j, k]
                )

    cosserat_internal_stress /= dilatation
    internal_forces[:] = difference_kernel_for_block_structure(
        cosserat_internal_stress, ghost_elems_idx
    )


@numba.njit(cache=True)  # type: ignore
def _compute_internal_torques(
    position_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    tangents: NDArray[np.float64],
    lengths: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    bend_matrix: NDArray[np.float64],
    rest_kappa: NDArray[np.float64],
    kappa: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
    mass_second_moment_of_inertia: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
    internal_stress: NDArray[np.float64],
    internal_couple: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    dilatation_rate: NDArray[np.float64],
    internal_torques: NDArray[np.float64],
    ghost_voronoi_idx: NDArray[np.int32],
) -> None:
    """
    Update <internal torque>.
    """
    # Compute \tau_l and cache it using internal_couple
    # Be careful about usage though
    _compute_internal_bending_twist_stresses_from_model(
        director_collection,
        rest_voronoi_lengths,
        internal_couple,
        bend_matrix,
        kappa,
        rest_kappa,
    )

    voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation**3

    # Delta(\tau_L / \Epsilon^3)
    bend_twist_couple_2D = difference_kernel_for_block_structure(
        internal_couple * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx
    )
    # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
    bend_twist_couple_3D = quadrature_kernel_for_block_structure(
        _batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths
        * voronoi_dilatation_inv_cube_cached,
        ghost_voronoi_idx,
    )
    # (Qt x n_L) * \hat{l}
    shear_stretch_couple = (
        _batch_cross(_batch_matvec(director_collection, tangents), internal_stress)
        * rest_lengths
    )

    blocksize = internal_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            internal_torques[i, k] = (
                bend_twist_couple_2D[i, k]
                + bend_twist_couple_3D[i, k]
                + shear_stretch_couple[i, k]
            )

    return bend_twist_couple_2D, shear_stretch_couple


@numba.njit(cache=True)  # type: ignore
def _compute_internal_torques_without_kappa_update(
    position_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    tangents: NDArray[np.float64],
    lengths: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    director_collection: NDArray[np.float64],
    rest_voronoi_lengths: NDArray[np.float64],
    bend_matrix: NDArray[np.float64],
    rest_kappa: NDArray[np.float64],
    kappa: NDArray[np.float64],
    voronoi_dilatation: NDArray[np.float64],
    mass_second_moment_of_inertia: NDArray[np.float64],
    omega_collection: NDArray[np.float64],
    internal_stress: NDArray[np.float64],
    internal_couple: NDArray[np.float64],
    dilatation: NDArray[np.float64],
    dilatation_rate: NDArray[np.float64],
    internal_torques: NDArray[np.float64],
    ghost_voronoi_idx: NDArray[np.int32],
) -> None:
    voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation**3

    # Delta(\tau_L / \Epsilon^3)
    bend_twist_couple_2D = difference_kernel_for_block_structure(
        internal_couple * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx
    )
    # \mathcal{A}[ (\kappa x \tau_L ) * \hat{D} / \Epsilon^3 ]
    bend_twist_couple_3D = quadrature_kernel_for_block_structure(
        _batch_cross(kappa, internal_couple)
        * rest_voronoi_lengths
        * voronoi_dilatation_inv_cube_cached,
        ghost_voronoi_idx,
    )
    # (Qt x n_L) * \hat{l}
    shear_stretch_couple = (
        _batch_cross(_batch_matvec(director_collection, tangents), internal_stress)
        * rest_lengths
    )

    blocksize = internal_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            internal_torques[i, k] = (
                bend_twist_couple_2D[i, k]
                + bend_twist_couple_3D[i, k]
                + shear_stretch_couple[i, k]
            )

    return bend_twist_couple_2D, shear_stretch_couple


@numba.njit(cache=True)  # type: ignore
def _update_accelerations(
    acceleration_collection: NDArray[np.float64],
    internal_forces: NDArray[np.float64],
    external_forces: NDArray[np.float64],
    mass: NDArray[np.float64],
    alpha_collection: NDArray[np.float64],
    inv_mass_second_moment_of_inertia: NDArray[np.float64],
    internal_torques: NDArray[np.float64],
    external_torques: NDArray[np.float64],
    dilatation: NDArray[np.float64],
) -> None:
    """
    Update <acceleration and angular acceleration> given <internal force/torque and external force/torque>.
    """

    blocksize_acc = internal_forces.shape[1]
    blocksize_alpha = internal_torques.shape[1]

    for i in range(3):
        for k in range(blocksize_acc):
            acceleration_collection[i, k] = (
                internal_forces[i, k] + external_forces[i, k]
            ) / mass[k]

    alpha_collection *= 0.0
    for i in range(3):
        for j in range(3):
            for k in range(blocksize_alpha):
                alpha_collection[i, k] += (
                    inv_mass_second_moment_of_inertia[i, j, k]
                    * (internal_torques[j, k] + external_torques[j, k])
                ) * dilatation[k]


@numba.njit(cache=True)  # type: ignore
def _zeroed_out_external_forces_and_torques(
    external_forces: NDArray[np.float64], external_torques: NDArray[np.float64]
) -> None:
    """
    This function is to zeroed out external forces and torques.

    Notes
    -----
    Microbenchmark results 100 elements
    python version: 3.32 µs ± 44.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    this version: 583 ns ± 1.94 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """
    n_nodes = external_forces.shape[1]
    n_elems = external_torques.shape[1]

    for i in range(3):
        for k in range(n_nodes):
            external_forces[i, k] = 0.0

    for i in range(3):
        for k in range(n_elems):
            external_torques[i, k] = 0.0
