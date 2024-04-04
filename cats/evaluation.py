import math

import numpy as np
import gymnasium as gym
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.cm as cm
from sklearn.neighbors import KernelDensity

from kitten.experience.memory import ReplayBuffer
from kitten.experience import Transitions

from cats import CatsExperiment


def entropy_memory(memory: ReplayBuffer):
    # Construct a density estimator
    s = Transitions(*memory.sample(len(memory))[0][:5]).s_0.cpu().numpy()
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(s)
    log_likelihoods = kde.score_samples(kde.sample(n_samples=10000))
    return -log_likelihoods.mean()


def visualise_teleport_targets(experiment: CatsExperiment, fig: Figure, ax: Axes):
    env = experiment.env
    s = np.array(experiment.logger._engine.results["teleport_targets_observations"])
    s, (x_label, x_low, x_high), (y_label, y_low, y_high) = env_to_2d(env, s)
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel(x_label)
    ax.set_ylim(y_low, y_high)
    ax.set_ylabel(y_label)
    for i, (x, y) in enumerate(s):
        ax.scatter(x, y, s=2)
    ax.set_title("Teleportation Targets")
    return fig, ax


def env_to_2d(env: gym.Env, s):
    supported_environments = ["MountainCarContinuous-v0", "Pendulum-v1"]
    if not env.spec.id in supported_environments:
        raise ValueError("Environment not supported")
    match env.spec.id:
        case "MountainCarContinuous-v0":
            return (
                s if s is not None else None,
                ("Position", env.observation_space.low[0], env.observation_space.high[0]),
                ("Velocity", env.observation_space.low[1], env.observation_space.high[1])
            )
        case "Pendulum-v1":
            return (
                np.arctan2(s[:, 1], s[:, 0]) if s is not None else None, 
                ("Theta", -math.pi, math.pi),
                ("Angular Velocity", env.observation_space.low[2], env.observation_space.high[2])
            )
        case _:
            raise ValueError("Environment not yet supported")


def visualise_memory(experiment: CatsExperiment, fig: Figure, ax: Axes):
    """Visualise state space for given environments"""

    env = experiment.env
    memory = experiment.memory.rb
    batch = Transitions(*memory.storage[:5])
    s = batch.s_0.cpu().numpy()

    ax.set_title("State Space Coverage")
    s, (x_label, x_low, x_high), (y_label, y_low, y_high) = env_to_2d(env, s)
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel(x_label)
    ax.set_ylim(y_low, y_high)
    ax.set_ylabel(y_label)

    # Colors based on time
    norm = mpl.colors.Normalize(vmin=0, vmax=len(s) - 1)
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(np.linspace(0, len(s) - 1, len(s)))
    ax.scatter(s[:, 0], s[:, 1], s=1, c=colors)

    fig.colorbar(m, ax=ax)

def visualise_experiment_value_estimate(
    experiment: CatsExperiment,
    fig: Figure,
    ax_v: Axes | None = None,
    ax_p: Axes | None = None
):
    device = experiment.device
    supported_environments = ["MountainCarContinuous-v0", "Pendulum-v1"]
    if not experiment.env.unwrapped.spec.id in supported_environments:
        raise ValueError("Environment not supported")
    # Plot

    # Construct Grid
    X = torch.linspace(
        experiment.env.observation_space.low[0],
        experiment.env.observation_space.high[0],
        100,
    )
    Y = torch.linspace(
        experiment.env.observation_space.low[1],
        experiment.env.observation_space.high[1],
        100,
    )
    grid_X, grid_Y = torch.meshgrid((X, Y))
    states = torch.stack((grid_X.flatten(), grid_Y.flatten())).T
    states = states.to(device)
    # Observation Normalisation
    states_cpu = states.cpu()
    states = experiment.rmv.transform(states)
    # V
    if ax_v is not None:
        values = experiment.algorithm.value.v(states)
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        cmap = cm.viridis
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(values.detach().cpu())

        ax_v.set_title("Value Function Visualisation")
        ax_v.scatter(states_cpu[:, 0], states_cpu[:, 1], c=colors)
        fig.colorbar(m, ax=ax_v)

    # Policy
    actions = experiment.algorithm.policy_fn(states)[:, :1]
    if ax_p is not None:
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.viridis
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(actions.detach().cpu())
        ax_p.scatter(states_cpu[:, 0], states_cpu[:, 1], c=colors)
        ax_p.set_title("Policy Actions")
        fig.colorbar(m, ax=ax_p)
