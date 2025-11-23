"""
Plotting functions for visualising simulations.
"""

# Import external modules.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Import QuEvolutio modules.
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RVectors,
    GVectors,
    RTensor,
    GTensor,
    GTensors,
)
from quevolutio.core.domain import QuantumHilbertSpace, TimeGrid

# Matplotlib settings.
mpl.rcParams["font.family"] = "Lato"
mpl.rcParams["mathtext.fontset"] = "cm"


def plot_state_1D(
    state: GVector, domain: QuantumHilbertSpace, title: str, filename: str
) -> None:
    """
    Plots the probability density, real component and complex component of a
    one-dimensional (1D) state.

    Parameters
    ----------
    state : GVector
        The one-dimensional (1D) state to plot. This should have shape
        (*domain.num_points).
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) that the state is defined on.
    title : str
        The title of the plot.
    filename : str
        The filename to which to save the figure to. This should contain the
        file extension.
    """

    # Calculate the probability density.
    prob_density: RVector = np.abs(state) ** 2

    # Create the figure.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the axis limits.
    ax.set_ylim(-1.0, 1.0)

    # Plot the probability density.
    ax.plot(
        domain.position_axes[0],
        prob_density,
        color="rebeccapurple",
        label=r"$|\psi(x)|^{2}$",
    )
    ax.fill_between(
        domain.position_axes[0],
        0.0,
        prob_density,
        color="rebeccapurple",
        alpha=0.50,
    )

    # Plot the real and complex components.
    ax.plot(
        domain.position_axes[0],
        state.real,
        label=r"$\mathrm{{Re}}[\psi(x)]$",
        linestyle="-",
        color="royalblue",
        alpha=0.75,
    )
    ax.plot(
        domain.position_axes[0],
        state.imag,
        label=r"$\mathrm{{Im}}[\psi(x)]$",
        linestyle="--",
        color="crimson",
        alpha=0.75,
    )

    # Set the axis labels.
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(
        r"$|\psi(x)|^{2}, \; \mathrm{{Re}}[\psi(x)], \; \mathrm{{Im}}[\psi(x)]$"
    )
    ax.legend(loc="upper right")

    # Save the figure.
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_state_2D(
    state: GTensor, domain: QuantumHilbertSpace, title: str, filename: str
) -> None:
    """
    Plots the probability density of a two-dimensional (2D) state on a heatmap.

    Parameters
    ----------
    state : GTensor
        The two-dimensional (2D) state to plot. This should have shape
        (*domain.num_points).
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) that the state is defined on.
    title : str
        The title of the plot.
    filename : str
        The filename to which to save the figure to. This should contain the
        file extension.
    """

    # Calculate the probability density.
    prob_density: RTensor = np.abs(state) ** 2

    # Create the figure.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the probability density.
    heatmap = ax.pcolormesh(
        domain.position_axes[0],
        domain.position_axes[1],
        prob_density.T,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(heatmap, ax=ax, label=r"$|\psi(x_{1}, x_{2})|^{2}$")

    # Set the axis labels.
    ax.set_title(title)
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")

    # Save the figure.
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def animate_states_1D(
    states: GVectors, domain: QuantumHilbertSpace, time_domain: TimeGrid, filename: str
) -> None:
    """
    Animates the probability density, real component and complex component of a
    set of one-dimensional (1D) states.

    Parameters
    ----------
    states : GVectors
        The one-dimensional (1D) states to animate. This should have shape
        (time_points, *domain.num_points).
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) that the states are defined on.
    time_domain : TimeGrid
        The time domain (grid) that the states are defined across.
    filename : str
        The filename to which to save the figure to. This should contain the
        file extension.

    Notes
    -----
    This function requires FFmpeg to be installed.
    """

    # Set the frames to animate.
    step: int = max(1, int(time_domain.num_points * 0.01))
    frames: range = range(0, time_domain.num_points, step)

    # Calculate the probability densities.
    states_prob_densities: RVectors = np.abs(states[frames]) ** 2

    # Store the real and complex components.
    states_real: RVectors = states[frames].real
    states_imag: RVectors = states[frames].imag

    # Create the figure.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the axis limits.
    ax.set_ylim(-1.0, 1.0)

    # Plot the probability density of the first state.
    (line_prob_density,) = ax.plot(
        domain.position_axes[0],
        states_prob_densities[0],
        color="rebeccapurple",
        label=r"$|\psi(x)|^{2}$",
    )
    fill_prob_density = ax.fill_between(
        domain.position_axes[0],
        0.0,
        states_prob_densities[0],
        color="rebeccapurple",
        alpha=0.50,
    )

    # Plot the real and complex components of the first state.
    (line_real,) = ax.plot(
        domain.position_axes[0],
        states_real[0],
        label=r"$\mathrm{{Re}}[\psi(x)]$",
        linestyle="-",
        color="royalblue",
        alpha=0.75,
    )
    (line_imag,) = ax.plot(
        domain.position_axes[0],
        states_imag[0],
        label=r"$\mathrm{{Im}}[\psi(x)]$",
        linestyle="--",
        color="crimson",
        alpha=0.75,
    )

    # Set the labels.
    title = ax.set_title(r"$\text{State} \; (T = 0.00)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(
        r"$|\psi(x)|^{2}, \; \mathrm{{Re}}[\psi(x)], \; \mathrm{{Im}}[\psi(x)]$"
    )
    ax.legend(loc="upper right")

    # Define the animation function.
    def animate(frame: int):
        nonlocal fill_prob_density

        # Update the lines.
        line_prob_density.set_ydata(states_prob_densities[frame])
        line_real.set_ydata(states_real[frame])
        line_imag.set_ydata(states_imag[frame])

        # Update the fill.
        fill_prob_density.remove()
        fill_prob_density = ax.fill_between(
            domain.position_axes[0],
            0.0,
            states_prob_densities[frame],
            color="rebeccapurple",
            alpha=0.50,
        )

        # Update the title.
        title.set_text(
            rf"$\text{{State}} \; (T = {time_domain.time_axis[frames[frame]]:.2f})$"
        )

        return line_prob_density, line_real, line_imag, fill_prob_density, title

    # Set the animation settings.
    fps: int = 30
    bitrate: int = 2000
    dpi: int = 200

    # Render the animation.
    ani = FuncAnimation(fig, animate, len(frames), blit=True, interval=(1000 / fps))
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)

    # Save the animation.
    ani.save(filename, writer, dpi=dpi)
    plt.close(fig)


def animate_states_2D(
    states: GTensors, domain: QuantumHilbertSpace, time_domain: TimeGrid, filename: str
) -> None:
    """
    Animates the probability density of a set of two-dimensional (2D) states on
    a heatmap.

    Parameters
    ----------
    states : GTensors
        The two-dimensional (2D) states to animate. This should have shape
        (time_points, *domain.num_points).
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) that the states are defined on.
    time_domain : TimeGrid
        The time domain (grid) that the states are defined across.
    filename : str
        The filename to which to save the figure to. This should contain the
        file extension.

    Notes
    -----
    This function requires FFmpeg to be installed.
    """

    # Set the frame to animate.
    step: int = max(1, int(time_domain.num_points * 0.01))
    frames: range = range(0, time_domain.num_points, step)

    # Calculate the probability densities.
    states_prob_densities: RVectors = np.abs(states[frames]) ** 2

    # Create the figure.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the probability density.
    heatmap = ax.pcolormesh(
        domain.position_axes[0],
        domain.position_axes[1],
        states_prob_densities[0].T,
        vmin=np.min(states_prob_densities),
        vmax=np.max(states_prob_densities),
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(heatmap, ax=ax, label=r"$|\psi(x_{1}, x_{2})|^{2}$")

    # Set the axis labels.
    title = ax.set_title(r"$\text{State} \; (T = 0.00)$")
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")

    # Define the animation function.
    def animate(frame: int):
        # Update the heatmap and title.
        heatmap.set_array(states_prob_densities[frame].T.ravel())
        title.set_text(
            rf"$\text{{State}} \; (T = {time_domain.time_axis[frames[frame]]:.2f})$"
        )

        return heatmap, title

    # Set the animation settings.
    fps: int = 30
    bitrate: int = 2000
    dpi: int = 200

    # Render the animation.
    ani = FuncAnimation(fig, animate, len(frames), blit=False, interval=(1000 / fps))
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)

    # Save the animation.
    ani.save(filename, writer, dpi=dpi)
    plt.close(fig)
