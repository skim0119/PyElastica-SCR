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
from .._so3 import exp_so3


class StableCosseratRod(CosseratRod):
    """ """

    REQUISITE_MODULES: list[Type] = []

    def __init__(
        self,
        *args: Any,
        num_gauss_siedel_iteration: int = 5,
        gauss_siedel_tolerence: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """
        The rotation update is done using a stable Gauss-Siedel method.

        Parameters
        ----------
        num_gauss_siedel_iteration: int, optional
            Number of Gauss-Siedel iterations to perform for rotation update. Default is 5.
        gauss_siedel_tolerence: float, optional
            Tolerance for Gauss-Siedel convergence. Default is 1e-6.

        """
        super().__init__(*args, **kwargs)
        self.num_gauss_siedel_iteration = num_gauss_siedel_iteration
        self.gauss_siedel_tolerence = gauss_siedel_tolerence

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


# Below is the numba-implementation of Cosserat Rod equations. They don't need to be visible by users.


def _compute_torque_residual(
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
):
    N_v = kappa.shape[1]
    residual = np.zeros((3, N_v))

    # Internal moments m = B * kappa
    m = np.einsum("ijk,jk->ik", bend_matrix, kappa)

    # Internal forces n = S * sigma
    n = np.einsum("ijk,jk->ik", shear_matrix, sigma)

    for j in range(N_v):
        # ---- m_s term (finite difference) ----
        if j == 0:
            m_s = (m[:, j + 1] - m[:, j]) / voronoi_length[j]
        elif j == N_v - 1:
            m_s = (m[:, j] - m[:, j - 1]) / voronoi_length[j]
        else:
            m_s = (m[:, j + 1] - m[:, j - 1]) / (2 * voronoi_length[j])

        # ---- kappa × m ----
        kappa_cross_m = np.cross(kappa[:, j], m[:, j])

        # ---- nu × n (shear-force coupling) ----
        # nu ≈ Q^T * (l * t)
        Q = director[:, :, j]
        nu = Q.T @ (dilatation[j] * tangent[:, j])
        nu_cross_n = np.cross(nu, n[:, j])

        # ---- external couple ----
        C = external_couple[:, j]

        residual[:, j] = m_s + kappa_cross_m + nu_cross_n + C

    return residual


def _orientation_relaxation_step(
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
    relaxation=1.0,
):
    residual = _compute_torque_residual(
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

    N_v = kappa.shape[1]

    for j in range(N_v):
        # Effective stiffness (local Hessian approximation)
        K = bend_matrix[:, :, j]

        # Solve K * delta_theta = -residual
        try:
            delta_theta = np.linalg.solve(K, -residual[:, j])
        except np.linalg.LinAlgError:
            continue  # skip ill-conditioned case

        delta_theta *= relaxation

        # Lie group update
        director[:, :, j] = director[:, :, j] @ exp_so3(delta_theta)

    return director


def _update_orientation(
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
    max_iters,
    tol,
) -> np.float64:
    for it in range(max_iters):
        old_director = director.copy()

        director = _orientation_relaxation_step(
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

        diff = np.linalg.norm(director - old_director)
        if diff < tol:
            break

    return diff


@numba.njit(cache=True)  # type: ignore
def _compute_geometry_from_state(
    position_collection: NDArray[np.float64],
    volume: NDArray[np.float64],
    lengths: NDArray[np.float64],
    tangents: NDArray[np.float64],
    radius: NDArray[np.float64],
) -> None:
    """
    Update <length, tangents, and radius> given <position and volume>.
    """
    # Compute eq (3.3) from 2018 RSOS paper

    # Note : we can use the two-point difference kernel, but it needs unnecessary padding
    # and hence will always be slower
    position_diff = position_difference_kernel(position_collection)
    # FIXME: Here 1E-14 is added to fix ghost lengths, which is 0, and causes division by zero error!
    lengths[:] = _batch_norm(position_diff) + 1e-14
    # _reset_scalar_ghost(lengths, ghost_elems_idx, 1.0)

    for k in range(lengths.shape[0]):
        tangents[0, k] = position_diff[0, k] / lengths[k]
        tangents[1, k] = position_diff[1, k] / lengths[k]
        tangents[2, k] = position_diff[2, k] / lengths[k]
        # resize based on volume conservation
        radius[k] = np.sqrt(volume[k] / lengths[k] / np.pi)


@numba.njit(cache=True)  # type: ignore
def _compute_all_dilatations(
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
    Update <dilatation and voronoi_dilatation>
    """
    _compute_geometry_from_state(position_collection, volume, lengths, tangents, radius)
    # Caveat : Needs already set rest_lengths and rest voronoi domain lengths
    # Put in initialization
    for k in range(lengths.shape[0]):
        dilatation[k] = lengths[k] / rest_lengths[k]

    # Compute eq (3.4) from 2018 RSOS paper
    # Note : we can use trapezoidal kernel, but it has padding and will be slower
    voronoi_lengths = position_average(lengths)

    # Compute eq (3.4) from 2018 RSOS paper
    for k in range(voronoi_lengths.shape[0]):
        voronoi_dilatation[k] = voronoi_lengths[k] / rest_voronoi_lengths[k]


@numba.njit(cache=True)  # type: ignore
def _compute_dilatation_rate(
    position_collection: NDArray[np.float64],
    velocity_collection: NDArray[np.float64],
    lengths: NDArray[np.float64],
    rest_lengths: NDArray[np.float64],
    dilatation_rate: NDArray[np.float64],
) -> None:
    """
    Update dilatation_rate given position, velocity, length, and rest_length
    """
    # self.lengths = l_i = |r^{i+1} - r^{i}|
    r_dot_v = _batch_dot(position_collection, velocity_collection)
    r_plus_one_dot_v = _batch_dot(
        position_collection[..., 1:], velocity_collection[..., :-1]
    )
    r_dot_v_plus_one = _batch_dot(
        position_collection[..., :-1], velocity_collection[..., 1:]
    )

    blocksize = lengths.shape[0]

    for k in range(blocksize):
        dilatation_rate[k] = (
            (r_dot_v[k] + r_dot_v[k + 1] - r_dot_v_plus_one[k] - r_plus_one_dot_v[k])
            / lengths[k]
            / rest_lengths[k]
        )


@numba.njit(cache=True)  # type: ignore
def _compute_shear_stretch_strains(
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
) -> None:
    """
    Update <shear/stretch(sigma)> given <dilatation, director, and tangent>.
    """

    # Quick trick : Instead of evaliation Q(et-d^3), use property that Q*d3 = (0,0,1), a constant
    _compute_all_dilatations(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        dilatation,
        rest_lengths,
        rest_voronoi_lengths,
        voronoi_dilatation,
    )

    z_vector = np.array([0.0, 0.0, 1.0]).reshape(3, -1)
    sigma[:] = dilatation * _batch_matvec(director_collection, tangents) - z_vector


@numba.njit(cache=True)  # type: ignore
def _compute_internal_shear_stretch_stresses_from_model(
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
) -> None:
    """
    Update <internal stress> given <shear matrix, sigma, and rest_sigma>.

    Linear force functional
    Operates on
    S : (3,3,n) tensor and sigma (3,n)
    """
    _compute_shear_stretch_strains(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
        director_collection,
        sigma,
    )
    internal_stress[:] = _batch_matvec(shear_matrix, sigma - rest_sigma)


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
    """

    # Compute n_l and cache it using internal_stress
    # Be careful about usage though
    _compute_internal_shear_stretch_stresses_from_model(
        position_collection,
        volume,
        lengths,
        tangents,
        radius,
        rest_lengths,
        rest_voronoi_lengths,
        dilatation,
        voronoi_dilatation,
        director_collection,
        sigma,
        rest_sigma,
        shear_matrix,
        internal_stress,
    )

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
    # Compute dilatation rate when needed, dilatation itself is done before
    # in internal_stresses
    _compute_dilatation_rate(
        position_collection, velocity_collection, lengths, rest_lengths, dilatation_rate
    )

    # FIXME: change memory overload instead for the below calls!
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

    # I apply common sub expression elimination here, as J w / e is used in both the lagrangian and dilatation
    # terms
    # TODO : the _batch_matvec kernel needs to depend on the representation of J, and should be coded as such
    J_omega_upon_e = (
        _batch_matvec(mass_second_moment_of_inertia, omega_collection) / dilatation
    )

    # (J \omega_L / e) x \omega_L
    # Warning : Do not do micro-optimization here : you can ignore dividing by dilatation as we later multiply by it
    # but this causes confusion and violates SRP
    lagrangian_transport = _batch_cross(J_omega_upon_e, omega_collection)

    # Note : in the computation of dilatation_rate, there is an optimization opportunity as dilatation rate has
    # a dilatation-like term in the numerator, which we cancel here
    # (J \omega_L / e^2) . (de/dt)
    unsteady_dilatation = J_omega_upon_e * dilatation_rate / dilatation

    blocksize = internal_torques.shape[1]
    for i in range(3):
        for k in range(blocksize):
            internal_torques[i, k] = (
                bend_twist_couple_2D[i, k]
                + bend_twist_couple_3D[i, k]
                + shear_stretch_couple[i, k]
                + lagrangian_transport[i, k]
                + unsteady_dilatation[i, k]
            )


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
