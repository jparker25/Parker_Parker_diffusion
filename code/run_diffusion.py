"""
run_diffusion.py

Script containing  funcions for simulating societies of individualists and altruists.

Authors: John E. Parker (2026)
"""

# import python libararies
import numpy as np
from agent import Agent


def redistribution(agents, epsilon=1e-4):
    """
    Redistributes wealth based on ideeology.

    Parameters
    ----------
    agents : list
        list of agent objects to redistribute wealth among
    epsilon : float
        (optional, 1e-4) small value to check if all bankrupt

    Returns
    -------
    agents : list
        list of agent objects to redistribute wealth among
    """

    # don't distribute wealth if all individualist
    if all([agents[i].ideology == "individualist" for i in range(len(agents))]):
        return agents

    # redistribute wealth if altruists
    elif all([agents[i].ideology == "altruist" for i in range(len(agents))]):

        # get alll wealth
        total_wealth = np.sum([agents[i].x[-1] for i in range(len(agents))])

        # if total wealth is below epsilon for full bankruptcy
        if total_wealth < epsilon:
            return agents

        # iterate through agents and redistribute wealth evenly
        for i in range(len(agents)):
            agents[i].x[-1] = total_wealth / len(agents)
            agents[i].bankrupt[-1] = False
        return agents


def simulate(x0, ideologies, N=2, T=1, mu=0, dt=0.001, sigma=0.3, epsilon=1e-4):
    """
    Simulates a society of agents undergoing stochastic modeling.

    Parameters
    ----------
    x0 : np.ndarray
        initial wealth amount for each agent
    ideologies : list
        string list of agent's ideology
    N : int
        (optional, 2) number of agents in the society
    T : float
        (optional, 1) total simulation time
    mu : float
        (optional, 0) initial drift value for agent's DDM model
    dt : float
        (optional, 0.001) time step for simulation
    sigma : float
        (optional, 0.3) volatility for noise for agent's DDM model
    epsilon : float
        (optional, 1e-4) value for collective bankruptcy

    Returns
    -------
    agents : list
        list of agent objects to redistribute wealth among
    alive : np.ndarray
        boolean matrix for agents alive at each time step
    """

    # create society as list of agents
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

    # boolean for total altruist society or not
    altruist_society = all(ideo == "altruist" for ideo in ideologies)

    # total number of steps
    n_steps = int(T // dt)

    # boolean alive matrix at each timestep
    alive = np.ones((n_steps, N), dtype=bool)

    # iterate through all time steps
    for t in range(n_steps - 1):

        # update all agents
        for i in range(N):
            agents[i].update()

        # check if any agent bankrupt, if so redistribute
        if any([agents[i].bankrupt[-1] for i in range(N)]):
            agents = redistribution(agents, epsilon=epsilon)

        # Check if collective bankruptcy for altruist society
        if altruist_society:
            total_wealth = sum([agents[i].x[-1] for i in range(N)])
            if total_wealth < epsilon:
                alive[t + 1 :, :] = False
                break

        # set alive count for current time step
        for i in range(N):
            alive[t + 1, i] = not agents[i].bankrupt[-1]

        # end simulation if everyone is bankrupt
        if all([agents[i].bankrupt[-1] for i in range(N)]):
            alive[t + 1 :, :] = False
            break
    return agents, alive
