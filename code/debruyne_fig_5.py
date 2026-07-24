# python libraries
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
import sys
import seaborn as sns
from scipy.special import erf, erfc, comb

# user modules
from run_diffusion import *
from helpers import *

# figure directory
fig_dir = "../figures"

# simulation setup
T = 1000
dt = 0.01
D = 1
trials = 1000
n_steps = int(T // dt)
time = np.arange(n_steps) * dt

#### general survival probability ##################
# (Eq 10 De Bruyne et al., 2021)
survival_prob_func = lambda x0,D,time : erf(x0/np.sqrt(4*D*time))

# (Eq 22 De Bruyne et al., 2021)
individual_prob_func = lambda x0,D,time,n : sum([comb(N,m)*(erfc(x0/np.sqrt(4*D*time))**m)*(erf(x0/np.sqrt(4*D*time))**(N-m)) for m in range(n)])


# set up figure 
fig, ax = plt.subplots(1,2,figsize=(8,3),dpi=300,tight_layout=True)
axes = [ax[i] for i in range(2)]


####### RUN WITH 4 AGENTS ####################
N = 4
mu = np.ones(N) * 0.0
sigma = np.ones(N) * np.sqrt(2)
x0 = np.ones(N) * 1

# determine altruist and individualist probs for N
altruist_surival_probability = survival_prob_func(x0[0],D/N,time[1:])
individualist_N_prob = individual_prob_func(x0[0],D,time[1:],N)
individualist_N_minus_1_prob = individual_prob_func(x0[0],D,time[1:],N-1)


#### run altruists ####
ideologies = ["altruist"] * N
alive_counts = np.zeros((n_steps, N))
S1_counts = np.zeros(n_steps)
S2_counts = np.zeros(n_steps)

for trial in range(trials):
    agents, alive = simulate(
        x0=x0, ideologies=ideologies, T=T, N=N, mu=mu, sigma=sigma, dt=dt
    )
    alive_counts += alive
    S1_counts += np.all(alive,axis=1)

S1_sim_altriust = S1_counts / trials
survival_prob = alive_counts / trials

#### run individualists ####
ideologies = ["individualist"] * N
alive_counts = np.zeros((n_steps, N))

S1_counts = np.zeros(n_steps)
S2_counts = np.zeros(n_steps)

for trial in range(trials):
    agents, alive = simulate(
        x0=x0, ideologies=ideologies, T=T, N=N, mu=mu, sigma=sigma, dt=dt
    )
    alive_counts += alive
    S1_counts += (np.sum(alive, axis=1) >= 2) #alive[:, 0] & alive[:, 1]
    S2_counts += np.any(alive,axis=1)

S1_sim_individual = S1_counts / trials
S2_sim_individual = S2_counts / trials
survival_prob = alive_counts / trials


# plot theory curves 
axes[0].plot(time[1:],altruist_surival_probability,"r",label="altruist theory")
axes[0].plot(time[1:],individualist_N_minus_1_prob,"b",label="individ N-1 theory")
axes[0].plot(time[1:],individualist_N_prob,"g",label="individ N theory")

# plot simulated surves 
axes[0].plot(time[1:], S1_sim_individual[1:], "b--", label="1st individ (sim)")
axes[0].plot(time[1:], S2_sim_individual[1:], "g--", label="2nd individ (sim)")
axes[0].plot(time[1:], S1_sim_altriust[1:], "r--", label="altruist (sim)")

# fix labels
axes[0].set_xlabel("Time",fontsize=8)
axes[0].set_ylabel("Survival Probability",fontsize=8)

# change scale
axes[0].set_xscale("log")
axes[0].set_yscale("log")

# adjust axes
axes[0].set_xlim([0.01,100])
axes[0].set_ylim([0.01,1.1])

# adjust labels and legend
axes[0].legend(fontsize=6,edgecolor="k")
axes[0].set_xlabel("t")
axes[0].set_ylabel("$S(x_0=1,t)$")



####### RUN WITH 16 AGENTS ####################
N = 16
mu = np.ones(N) * 0.0
sigma = np.ones(N) * np.sqrt(2)
x0 = np.ones(N) * 1

# determine altruist and individualist probs for N
altruist_surival_probability = survival_prob_func(x0[0],D/N,time[1:])
individualist_N_prob = individual_prob_func(x0[0],D,time[1:],N)
individualist_N_minus_1_prob = individual_prob_func(x0[0],D,time[1:],N-1)


#### run altruists ####
ideologies = ["altruist"] * N
alive_counts = np.zeros((n_steps, N))
S1_counts = np.zeros(n_steps)
S2_counts = np.zeros(n_steps)

for trial in range(trials):
    agents, alive = simulate(
        x0=x0, ideologies=ideologies, T=T, N=N, mu=mu, sigma=sigma, dt=dt
    )
    alive_counts += alive
    S1_counts += np.all(alive,axis=1)

S1_sim_altriust = S1_counts / trials
survival_prob = alive_counts / trials

#### run individualists ####
ideologies = ["individualist"] * N
alive_counts = np.zeros((n_steps, N))

S1_counts = np.zeros(n_steps)
S2_counts = np.zeros(n_steps)

for trial in range(trials):
    agents, alive = simulate(
        x0=x0, ideologies=ideologies, T=T, N=N, mu=mu, sigma=sigma, dt=dt
    )
    alive_counts += alive
    S1_counts += (np.sum(alive, axis=1) >= 2) #alive[:, 0] & alive[:, 1]
    S2_counts += np.any(alive,axis=1)

S1_sim_individual = S1_counts / trials
S2_sim_individual = S2_counts / trials
survival_prob = alive_counts / trials


# plot theory curves 
axes[1].plot(time[1:],altruist_surival_probability,"r",label="altruist theory")
axes[1].plot(time[1:],individualist_N_minus_1_prob,"b",label="individ N-1 theory")
axes[1].plot(time[1:],individualist_N_prob,"g",label="individ N theory")

# plot simulated surves 
axes[1].plot(time[1:], S1_sim_individual[1:], "b--", label="1st individ (sim)")
axes[1].plot(time[1:], S2_sim_individual[1:], "g--", label="2nd individ (sim)")
axes[1].plot(time[1:], S1_sim_altriust[1:], "r--", label="altruist (sim)")

# fix labels
axes[1].set_xlabel("Time",fontsize=8)
axes[1].set_ylabel("Survival Probability",fontsize=8)

# change scale
axes[1].set_xscale("log")
axes[1].set_yscale("log")

# adjust axes
axes[1].set_xlim([0.01,1000])
axes[1].set_ylim([0.01,1.1])

# adjust labels and legend
axes[1].legend(fontsize=6,edgecolor="k")
axes[1].set_xlabel("t")
axes[1].set_ylabel("$S(x_0=1,t)$")


### clean up and save figure ###
makeNice(axes)
fig.savefig(f"{fig_dir}/debruyne_fig_5.pdf",bbox_inches="tight")
plt.close()

### open figure 
run_cmd(["open",f"{fig_dir}/debruyne_fig_5.pdf"])