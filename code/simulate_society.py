"""
run_society.py

Script to run any society containing a mix of agents.

Authors: John E. Parker (2026)
"""

# python libraries
import numpy as np
import json

# user modules
from run_diffusion import *
from helpers import *


def determine_gini_coefficient(wealth):
    """
    Compute the Gini coefficient given list of wealth.
    0 is perfect equality and 1 is perfect inequality.

    Parameters
    ----------
    wealth : list
        list of wealth values from agents

    Returns
    -------
    gini : float
        Gini coefficient
    """
    # gather total wealth
    wealth = np.array(wealth)
    total_wealth = np.sum(wealth)

    # if no wealth, perfect equality
    if total_wealth <= 0:
        return 0

    # calculate gini coefficient
    gini = 0
    for i in range(wealth.shape[0]):
        for j in range(wealth.shape[0]):
            gini += abs(wealth[i] - wealth[j])
    gini /= 2 * wealth.shape[0] * total_wealth
    return gini


def run_mixed_society_trials(
    N=4,
    mu=np.zeros(4),
    sigma=np.ones(4) * np.sqrt(2),
    x0=np.ones(4),
    redistribution_contract=1,
    fraction_alts=0.5,
    trials=1000,
    T=1000,
    dt=0.01,
    save_dir="../data",
    save_files=True,
    save_agents=False,
):
    """
    Run a mixed society of individualists and altruists.

    Parameters
    ----------
    N : int
        (optional, 4) number of agents to simulate
    mu : np.ndarray
        (optional, [0,0,0,0]) array of initial drift terms
    sigma : np.ndarray
        (optional, [sqrt(2),sqrt(2),sqrt(2),sqrt(2)]) array of volatility
    x0 : np.ndarray
        (optional, [1,1,1,1]) array of initial wealth values
    redistribution_contract : int
        (optional, 1) 1, 2, or 3 designating different social contracts
    fraction_alts : float
        (optional, 0.5) fraction of altruists in society
    trials : int
        (optional, 1000) number of trials to simulate for the society
    T : float
        (optional, 1000) final simulation time, if reached
    dt : float
        (optional, 0.01) time step for simulation
    save_dir : str
        (optional, ../data) path to store simulation files if saving
    save_files : bool
        (optional, True) saves files if true
    save_agents : bool
        (optional, False) saves agent wealth and bankruptcy if true

    Returns
    -------
    results : dict
        "S1_sim_individual": N-1 ind survival probability
        "S2_sim_individual": N ind survival probability
        "S1_sim_altriust": alt survival probability
        "S_sim_all": all survival probability,
        "ideologies": ideologies,
        "t_collective_ruin": time to collective ruin,
        "gini_all": Gini coefficient for society,
        "gini_alt": Gini coefficient for altruists,
        "gini_ind": Gini coefficient for individualists,
    """

    # set up time for simulation
    n_steps = int(T // dt)

    # mixed society ideologies
    ideologies = ["altruist"] * int(fraction_alts * N) + ["individualist"] * (
        N - int(fraction_alts * N)
    )

    # things to track
    alive_counts = np.zeros((n_steps, N))
    S1_ind_counts = np.zeros(n_steps)  # N - 1 individualist survival
    S2_ind_counts = np.zeros(n_steps)  # N individualist
    S1_alt_counts = np.zeros(n_steps)  # altruist
    S_all_counts = np.zeros(n_steps)  # society survival
    t_collective_ruin = np.zeros(trials)  # time to collective ruin
    gini_all = np.zeros(n_steps)  # gini coefficient for all
    gini_alt = np.zeros(n_steps)  # gini coefficient for altruists
    gini_ind = np.zeros(n_steps)  # gini coefficient for individdualists

    # determine indicies for each society ideology
    alt_indices = [i for i, ideo in enumerate(ideologies) if ideo == "altruist"]
    ind_indices = [i for i, ideo in enumerate(ideologies) if ideo == "individualist"]

    # simulate multiple trials
    for trial in range(trials):
        agents, alive = simulate(
            x0=x0,
            ideologies=ideologies,
            T=T,
            N=N,
            mu=mu,
            sigma=sigma,
            dt=dt,
            redistribution_contract=redistribution_contract,
        )

        # compute gini coefficients
        for t in range(n_steps):

            # entire society
            alive_mask = alive[t, :]
            if np.sum(alive_mask) > 1:
                wealth_t = [agents[i].x[t] for i in range(N) if alive_mask[i]]
                gini_all[t] += determine_gini_coefficient(wealth_t) / trials

            # individualists
            ind_alive_mask = alive_mask[ind_indices]
            if np.sum(ind_alive_mask) > 1:
                wealth_t = [agents[i].x[t] for i in range(N) if alive_mask[i]]
                gini_ind[t] += determine_gini_coefficient(wealth_t) / trials

            # altruists
            alt_alive_mask = alive_mask[alt_indices]
            if np.sum(alt_alive_mask) > 1:
                wealth_t = [agents[i].x[t] for i in range(N) if alive_mask[i]]
                gini_alt[t] += determine_gini_coefficient(wealth_t) / trials

        # save agent data if desired
        if save_agents:
            run_cmd(["mkdir", "-p", f"{save_dir}/trial_{trial:03d}"], print_out=False)
            for i, agent in enumerate(agents):
                np.savetxt(
                    f"{save_dir}/trial_{trial:03d}/agent_x_{i:03d}.txt",
                    agent.x,
                    delimiter=" ",
                    newline="\n",
                    fmt="%f",
                )
                np.savetxt(
                    f"{save_dir}/trial_{trial:03d}/agent_bankrupt_{i:03d}.txt",
                    agent.bankrupt,
                    delimiter=" ",
                    newline="\n",
                    fmt="%f",
                )

        # determine time to collective ruin
        alive_counts += alive
        all_dead = np.all(~alive, axis=1)
        if np.any(all_dead):
            t_ruin = np.argmax(all_dead)
            t_collective_ruin[trial] = t_ruin * dt
        else:
            t_collective_ruin[trial] = T

        # split alive data by individualists and altruists
        alive_ind = alive[:, ind_indices]
        alive_alt = alive[:, alt_indices]

        # determine trial proabilities and add to total
        S1_ind_counts += np.sum(alive_ind, axis=1) >= 2
        S2_ind_counts += np.any(alive_ind, axis=1)
        S1_alt_counts += np.all(alive_alt, axis=1)
        S_all_counts += np.all(alive, axis=1)

    # determine individualist N - 1 survival probabilities
    S1_sim_individual = S1_ind_counts / trials

    # determine individualist N survival probabilities
    S2_sim_individual = S2_ind_counts / trials

    # determine altruist survival probability
    S1_sim_altriust = S1_alt_counts / trials

    # determine society survival probability
    S_sim_all = S_all_counts / trials

    # dictionary containing main results
    results = {
        "S1_sim_individual": S1_sim_individual,
        "S2_sim_individual": S2_sim_individual,
        "S1_sim_altriust": S1_sim_altriust,
        "S_sim_all": S_sim_all,
        "ideologies": ideologies,
        "t_collective_ruin": t_collective_ruin,
        "gini_all": gini_all,
        "gini_alt": gini_alt,
        "gini_ind": gini_ind,
    }

    # save results and metadata if desired
    if save_files:
        run_cmd(["mkdir", "-p", f"{save_dir}/trial_{trial:03d}"], print_out=False)
        meta_data = {
            "N": N,
            "mu": list(mu),
            "sigma": list(sigma),
            "x0": list(x0),
            "fraction_alts": fraction_alts,
            "trials": trials,
            "T": T,
            "dt": dt,
            "ideologies": ideologies,
        }
        with open(f"{save_dir}/meta_data.json", "w") as f:
            json.dump(meta_data, f)
        np.savetxt(
            f"{save_dir}/S_sim_all.txt",
            S_sim_all,
            fmt="%f",
            delimiter=" ",
            newline="\n",
        )
        np.savetxt(
            f"{save_dir}/S1_sim_altriust.txt",
            S1_sim_altriust,
            fmt="%f",
            delimiter=" ",
            newline="\n",
        )
        np.savetxt(
            f"{save_dir}/S2_sim_individual.txt",
            S2_sim_individual,
            fmt="%f",
            delimiter=" ",
            newline="\n",
        )
        np.savetxt(
            f"{save_dir}/S1_sim_individual.txt",
            S1_sim_individual,
            fmt="%f",
            delimiter=" ",
            newline="\n",
        )
        np.savetxt(
            f"{save_dir}/t_collective_ruin.txt",
            t_collective_ruin,
            fmt="%f",
            delimiter=" ",
            newline="\n",
        )
        np.savetxt(
            f"{save_dir}/gini_all.txt",
            gini_all,
            fmt="%f",
            delimiter=" ",
            newline="\n",
        )
        np.savetxt(
            f"{save_dir}/gini_alt.txt",
            gini_alt,
            fmt="%f",
            delimiter=" ",
            newline="\n",
        )
        np.savetxt(
            f"{save_dir}/gini_ind.txt",
            gini_ind,
            fmt="%f",
            delimiter=" ",
            newline="\n",
        )
    return results
