import math

import numpy as np
import gymnasium as gym
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
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


def visualise_teleport_targets(experiment: CatsExperiment):
    fig, ax = plt.subplots()
    env = experiment.env
    ax.set_xlim(env.observation_space.low[0], env.observation_space.high[0])
    ax.set_xlabel("Position")
    ax.set_ylim(env.observation_space.low[1], env.observation_space.high[1])
    ax.set_ylabel("Velocity")
    for x, y in experiment.log["teleport_targets_observations"]:
        ax.scatter(x, y)
    return fig, ax


def visualise_memory(experiment: CatsExperiment):
    """Visualise state space for given environmentss"""

    env = experiment.env
    memory = experiment.memory.rb

    fig, ax = plt.subplots()
    ax.set_title("State Space Coverage")

    if env.spec.id == "MountainCarContinuous-v0":
        ax.set_xlim(env.observation_space.low[0], env.observation_space.high[0])
        ax.set_xlabel("Position")
        ax.set_ylim(env.observation_space.low[1], env.observation_space.high[1])
        ax.set_ylabel("Velocity")
    elif env.spec.id == "Pendulum-v1":
        ax.set_xlim(-math.pi, math.pi)
        ax.set_xlabel("Theta")
        ax.set_ylim(env.observation_space.low[2], env.observation_space.high[2])
        ax.set_ylabel("Angular Velocity")

    batch = Transitions(*memory.storage[:5])
    s = batch.s_0.cpu().numpy()
    # Colors based on time
    norm = mpl.colors.Normalize(vmin=0, vmax=len(s) - 1)
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(np.linspace(0, len(s) - 1, len(s)))
    if env.spec.id == "MountainCarContinuous-v0":
        ax.scatter(s[:, 0], s[:, 1], s=1, c=colors)
    elif env.spec.id == "Pendulum-v1":
        ax.scatter(np.arctan2(s[:, 1], s[:, 0]), s[:, 2], s=1, c=colors)

    fig.colorbar(m, ax=ax)


def visualise_experiment_value_estimate(
    experiment: CatsExperiment, device: str = "cpu"
):
    supported_environments = ["MountainCarContinuous-v0"]
    if not experiment.env.unwrapped.spec.id in supported_environments:
        raise ValueError("Environment not supported")
    # Plot
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)

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
    states = experiment.normalise_observation.transform(states)
    # V
    ax = axs[0]
    values = experiment.V(states)
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(values.detach().cpu())

    ax.set_title("Value Function Visualisation")
    ax.scatter(states_cpu[:, 0], states_cpu[:, 1], c=colors)
    fig.colorbar(m, ax=ax)

    # Policy
    ax = axs[1]
    actions = experiment.algorithm.policy_fn(states)[:, :1]
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(actions.detach().cpu())
    ax.scatter(states_cpu[:, 0], states_cpu[:, 1], c=colors)
    ax.set_title("Policy Actions")
    fig.colorbar(m, ax=ax)
