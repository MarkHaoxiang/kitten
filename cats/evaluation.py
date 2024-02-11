from sklearn.neighbors import KernelDensity

from curiosity.experience.memory import ReplayBuffer
from curiosity.experience import Transition

def entropy_memory(memory: ReplayBuffer):
    # Construct a density estimator
    s = Transition(*memory.sample(len(memory))[0]).s_0.cpu().numpy()
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(s)
    log_likelihoods = kde.score_samples(kde.sample(n_samples=10000))
    return -log_likelihoods.mean()