import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
import sys
import seaborn as sns


def redistribution(agents, epsilon=1e-4):
    if all([agents[i].ideology == "individualist" for i in range(len(agents))]):
        return agents
    elif all([agents[i].ideology == "altruist" for i in range(len(agents))]):
        total_wealth = np.sum([agents[i].x[-1] for i in range(len(agents))])
        if total_wealth < epsilon:
            return agents
        for i in range(len(agents)):
            agents[i].x[-1] = total_wealth / len(agents)
            agents[i].bankrupt[-1] = False
        return agents


def simulate(x0, ideologies, N=2, T=1, mu=0, dt=0.001, sigma=0.3, epsilon=1e-4):
    agents = [
        Agent(
            x0=x0[i],
            mu=mu[i],
            dt=dt,
            sigma=sigma[i],
            ideology=ideologies[i],
            seed=np.random.randint(0, 1_000_000),
        )
        for i in range(N)
    ]
    xs = np.zeros((int(T // dt), N))
    xs[0] = x0
    altruist_society = all(ideo == "altruist" for ideo in ideologies)

    n_steps = int(T // dt)
    alive = np.ones((n_steps, N), dtype=bool)
    for t in range(xs.shape[0] - 1):

        for i in range(N):
            agents[i].update()
        if any([agents[i].bankrupt[-1] for i in range(N)]):
            agents = redistribution(agents, epsilon=epsilon)

        if altruist_society:
            total_wealth = sum([agents[i].x[-1] for i in range(N)])
            if total_wealth < epsilon:
                alive[t + 1 :, :] = False
                break

        for i in range(N):
            alive[t + 1, i] = not agents[i].bankrupt[-1]

        if all([agents[i].bankrupt[-1] for i in range(N)]):
            alive[t + 1 :, :] = False
            break
    return agents, alive
