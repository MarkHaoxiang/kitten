import torch.nn as nn
import gymnasium as gym


ENVIRONMENT = 'CartPole-v1'
FEATURES = 128

def train():
    critic = nn.Sequential(
        nn.LazyLinear(out_features=128),
        nn.Tanh(),
        nn.Linear(in_features=FEATURES, out_features=FEATURES),
        nn.Tanh(),
        nn.Linear(in_features=FEATURES, out_features=FEATURES),
        nn.LeakyReLU()
    )
if __name__ == "__main__":
    train()