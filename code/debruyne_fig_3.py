# python libraries
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
import sys
import seaborn as sns
from scipy.special import erf

# user modules
from run_diffusion import *
from helpers import *

# figure directory
fig_dir = "../figures"

# simulation setup
T = 1000
dt = 0.01
N = 2
mu = np.ones(N) * 0.0
sigma = np.ones(N) * np.sqrt(2)
x0 = np.ones(N) * 1
trials = 1000
n_steps = int(T // dt)
time = np.arange(n_steps) * dt
D = 1

survival_prob_func = lambda x0,D,time : erf(x0/np.sqrt(4*D*time))

altruist_surival_probability = survival_prob_func(x0[0],D/2,time[1:])
first_ind_surival_probability = survival_prob_func(x0[0],D,time[1:])**2
sec_ind_survival_probability=first_ind_surival_probability + 2*(1-survival_prob_func(x0[0],D,time[1:]))*(survival_prob_func(x0[0],D,time[1:]))

#### run two individualists ####
ideologies = ["individualist"] * N
alive_counts = np.zeros((n_steps, N))

S1_counts = np.zeros(n_steps)
S2_counts = np.zeros(n_steps)

for trial in range(trials):
    agents, alive = simulate(
        x0=x0, ideologies=ideologies, T=T, N=N, mu=mu, sigma=sigma, dt=dt
    )
    alive_counts += alive
    S1_counts += alive[:, 0] & alive[:, 1]
    S2_counts += alive[:, 0] | alive[:, 1]

S1_sim_individual = S1_counts / trials
S2_sim_individual = S2_counts / trials
survival_prob = alive_counts / trials

#### run two altruists ####
ideologies = ["altruist"] * N
alive_counts = np.zeros((n_steps, N))
S1_counts = np.zeros(n_steps)
S2_counts = np.zeros(n_steps)

for trial in range(trials):
    agents, alive = simulate(
        x0=x0, ideologies=ideologies, T=T, N=N, mu=mu, sigma=sigma, dt=dt
    )
    alive_counts += alive
    S1_counts += alive[:, 0] & alive[:, 1]

S1_sim_altriust = S1_counts / trials
survival_prob = alive_counts / trials

# plot
fig, ax = plt.subplots(1,1,figsize=(4,3),dpi=300,tight_layout=True)
ax.plot(time[1:],altruist_surival_probability,"r",label="altruist Theory")
ax.plot(time[1:],first_ind_surival_probability,"b",label="1st individ Theory")
ax.plot(time[1:],sec_ind_survival_probability,"g",label="2nd individ Theory")

ax.plot(time[1:], S1_sim_individual[1:], "b--", label="1st individ (sim)")

ax.plot(time[1:], S2_sim_individual[1:], "g--", label="2nd individ (sim)")

ax.plot(time[1:], S1_sim_altriust[1:], "r--", label="altruist (sim)")

ax.set_xlabel("Time",fontsize=8)
ax.set_ylabel("Survival Probability",fontsize=8)
ax.set_ylim([0, 1])
ax.set_xlim([0, 10])

ax.legend(fontsize=6,edgecolor="k")
ax.set_xlabel("t")
ax.set_ylabel("$S(x_0=1,t)$")
makeNice(ax)
fig.savefig(f"{fig_dir}/debruyne_fig_3.pdf",bbox_inches="tight")
plt.close()

run_cmd(["open",f"{fig_dir}/debruyne_fig_3.pdf"])

