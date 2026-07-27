"""
run_diffusion.py

Script containing  funcions for simulating societies of individualists and altruists.

Authors: John E. Parker (2026)
"""

# import python libararies
import numpy as np
from agent import Agent


def redistribution(agents, epsilon=1e-4, social_contract=0):
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
        return alt_share_among_themselves(agents, epsilon=epsilon)

    # redistribute wealth according to social contract
    else:
        ## altruists share with everyone
        if social_contract == 1:
            return alt_share_with_everyone(agents, epsilon=epsilon)

        ## altruists share only among themselves
        elif social_contract == 2:
            return alt_share_among_themselves(agents, epsilon=epsilon)

        ## redistribution only occurs if altruist bankrupt
        elif social_contract == 3:
            if any(
                [agent.bankrupt[-1] for agent in agents if agent.ideology == "altruist"]
            ):
                return alt_share_among_themselves(agents, epsilon=epsilon)
            else:
                return agents


def alt_share_with_everyone(agents, epsilon=1e-4):
    """
    Social contract where altruists share with bankrupt agents, despite ideology.

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
    # gather altruist indices and find total wealth
    alt_indices = [i for i, agent in enumerate(agents) if agent.ideology == "altruist"]
    total_alt_wealth = np.sum([agents[i].x[-1] for i in alt_indices])

    # determine agents who are bankrupt
    num_bankrupt = [
        i
        for i, agent in enumerate(agents)
        if agent.bankrupt[-1] and agent.ideology == "individualist"
    ]

    # add to altruist wealth any bankrupt individualist agents
    wealth_to_redistribute = total_alt_wealth + np.sum(
        [
            agent.x[-1]
            for agent in agents
            if agent.bankrupt[-1] and agent.ideology == "individualist"
        ]
    )

    # Not enough money to redistribute
    if total_alt_wealth < epsilon:
        return agents

    # iterate through agents and redistribute wealth evenly
    for i in range(len(agents)):
        if agents[i].bankrupt[-1] or agents[i].ideology == "altruist":
            agents[i].x[-1] = wealth_to_redistribute / (
                len(alt_indices) + len(num_bankrupt)
            )
            agents[i].bankrupt[-1] = False
    return agents


def alt_share_among_themselves(agents, epsilon=1e-4):
    """
    Social contract where altruists share amongst themselves.

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
    # gather altruist indices and find total wealth
    alt_indices = [i for i, agent in enumerate(agents) if agent.ideology == "altruist"]
    total_alt_wealth = np.sum([agents[i].x[-1] for i in alt_indices])

    # Not enough money to redistribute
    if total_alt_wealth < epsilon:
        return agents

    # iterate through agents and redistribute wealth evenly
    for i in alt_indices:
        agents[i].x[-1] = total_alt_wealth / (len(alt_indices))
        agents[i].bankrupt[-1] = False
    return agents


def simulate(
    x0,
    ideologies,
    N=2,
    T=1,
    mu=0,
    dt=0.001,
    sigma=0.3,
    epsilon=1e-4,
    redistribution_contract=0,
):
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

    # validation
    valid_ideologies = ["altruist", "individualist"]
    if not all(ideo in valid_ideologies for ideo in ideologies):
        raise ValueError(
            f"Incorrect ideologies! Must be {valid_ideologies[0]} or {valid_ideologies[1]}."
        )

    # determine type of society
    altruist_society = all(ideo == "altruist" for ideo in ideologies)
    individualist_society = all(ideo == "individualist" for ideo in ideologies)
    mixed_society = not altruist_society and not individualist_society

    # check correct social contract
    if mixed_society and redistribution_contract not in [1, 2, 3]:
        raise ValueError(f"Incorrect mixed society contracts! Must be 1, 2, or 3.")

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
            agents = redistribution(
                agents, epsilon=epsilon, social_contract=redistribution_contract
            )

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
