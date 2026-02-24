import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb


def analytical_shearable(arg_rod, arg_end_force, n_elem=500):
    base_length = np.sum(arg_rod.rest_lengths)
    arg_s = np.linspace(0.0, base_length, n_elem)
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force
    acting_force = np.abs(acting_force)

    linear_prefactor = -acting_force / arg_rod.shear_matrix[0, 0, 0]
    quadratic_prefactor = (
        -acting_force
        * np.sum(arg_rod.rest_lengths)
        / 2.0
        / arg_rod.bend_matrix[0, 0, 0]
    )
    cubic_prefactor = acting_force / 6.0 / arg_rod.bend_matrix[0, 0, 0]
    return (
        arg_s,
        arg_s
        * (linear_prefactor + arg_s * (quadratic_prefactor + arg_s * cubic_prefactor)),
    )


def analytical_unshearable(arg_rod, arg_end_force, n_elem=500):
    base_length = np.sum(arg_rod.rest_lengths)
    arg_s = np.linspace(0.0, base_length, n_elem)
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force
    acting_force = np.abs(acting_force)

    quadratic_prefactor = (
        -acting_force
        * np.sum(arg_rod.rest_lengths)
        / 2.0
        / arg_rod.bend_matrix[0, 0, 0]
    )
    cubic_prefactor = acting_force / 6.0 / arg_rod.bend_matrix[0, 0, 0]
    return arg_s, arg_s**2 * (quadratic_prefactor + arg_s * cubic_prefactor)


def plot_timoshenko(
    rod,
    end_force,
    SAVE_FIGURE,
    ADD_UNSHEARABLE_ROD=False,
    recorded_history=None,
):
    def _plot_tangents(
        axis,
        position,
        director,
        every=5,
        color=to_rgb("xkcd:orange"),
        alpha=0.9,
        add_label=False,
    ):
        tangents = director[2, ...]
        segment_centers = 0.5 * (position[:, 1:] + position[:, :-1])
        sampled_idx = np.arange(0, tangents.shape[1], every)
        axis.quiver(
            segment_centers[2, sampled_idx],
            segment_centers[0, sampled_idx],
            tangents[2, sampled_idx],
            tangents[0, sampled_idx],
            color=color,
            alpha=alpha,
            scale=20.0,
            width=0.004,
            label="tangent (every 5 elems)" if add_label else None,
        )

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.grid(which="minor", color="k", linestyle="--")
    ax.grid(which="major", color="k", linestyle="-")
    analytical_shearable_position = analytical_shearable(rod, end_force)
    ax.plot(
        analytical_shearable_position[0],
        analytical_shearable_position[1],
        "k--",
        label="Timoshenko",
    )
    if recorded_history and recorded_history.get("position"):
        times = recorded_history.get("time", [])
        positions = recorded_history.get("position", [])
        directors = recorded_history.get("director", [])
        n_samples = len(positions)
        for i, (sample_time, position, director) in enumerate(
            zip(times, positions, directors)
        ):
            alpha = 0.3 + 0.7 * (i + 1) / max(n_samples, 1)
            ax.plot(
                position[2, ...],
                position[0, ...],
                c=to_rgb("xkcd:bluish"),
                alpha=alpha,
                lw=1.2,
                label=f"t={sample_time:.1f}s" if i == n_samples - 1 else None,
            )
            _plot_tangents(
                ax,
                position,
                director,
                every=5,
                alpha=alpha,
                add_label=i == n_samples - 1,
            )
    else:
        current_position = rod.position_collection
        ax.plot(
            rod.position_collection[2, ...],
            rod.position_collection[0, ...],
            c=to_rgb("xkcd:bluish"),
            label="n=" + str(rod.n_elems),
        )
        _plot_tangents(
            ax,
            current_position,
            rod.director_collection,
            every=5,
            add_label=True,
        )
    analytical_unshearable_position = analytical_unshearable(rod, end_force)
    ax.plot(
        analytical_unshearable_position[0],
        analytical_unshearable_position[1],
        "r-.",
        label="Euler-Bernoulli",
    )
    fig.legend()
    plt.show()
    if SAVE_FIGURE:
        fig.savefig("Timoshenko_beam_test" + str(rod.n_elems) + ".png")

    if recorded_history and recorded_history.get("position"):
        times = np.asarray(recorded_history.get("time", []), dtype=float)
        positions = recorded_history.get("position", [])
        tip_positions = np.asarray(
            [position[:, -1] for position in positions], dtype=float
        )

        fig_tip = plt.figure(figsize=(10, 6), frameon=True, dpi=150)
        ax_tip = fig_tip.add_subplot(111)
        ax_tip.grid(which="minor", color="k", linestyle="--")
        ax_tip.grid(which="major", color="k", linestyle="-")
        ax_tip.plot(times, tip_positions[:, 0], c=to_rgb("xkcd:bluish"), label="tip x")
        ax_tip.set_xlabel("time [s]")
        ax_tip.set_ylabel("tip x position [m]")
        ax_tip.set_title("Tip X Position Over Time")
        ax_tip.legend()
        plt.show()
        if SAVE_FIGURE:
            fig_tip.savefig("Timoshenko_tip_position_over_time.png")
