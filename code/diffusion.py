import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
import sys
import seaborn as sns


def redistribution(agents,dist_frac=0.5,epsilon = 1e-4):
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

def simulate(x0, ideologies,N=2, T=1, mu=0, dt=0.001, sigma=0.3):
    agents = [
        Agent(
            x0=x0[i], mu=mu[i], dt=dt, sigma=sigma[i],ideology=ideologies[i], seed=np.random.randint(0, 1_000_000)
        )
        for i in range(N)
    ]
    xs = np.zeros((int(T // dt), N))
    xs[0] = x0

    n_steps = int(T // dt)
    alive = np.ones((n_steps, N), dtype=bool)
    survival_rate = np.zeros(int(T // dt)) / N
    survival_rate[0] = 1
    for t in range(xs.shape[0] - 1):
        
        for i in range(N):
            agents[i].update()
        if any([agents[i].bankrupt[-1] for i in range(N)]):
            agents = redistribution(agents)

        for i in range(N):
            alive[t+1, i] = not agents[i].bankrupt[-1]

        if all([agents[i].bankrupt[-1] for i in range(N)]):
            survival_rate[t + 1] = 1 - np.sum([agents[i].bankrupt[-1] for i in range(N)]) / N
            survival_rate[t+1:] = 0.0
            alive[t+1:, :] = False
            break
        survival_rate[t + 1] = 1 - np.sum([agents[i].bankrupt[-1] for i in range(N)]) / N
    return agents, alive


T = 1000
dt = 0.01
N = 2
mu = np.ones(N) * 0.0
sigma = np.ones(N) * np.sqrt(2)
x0 = np.ones(N) * 1
ideologies = ["altruist"]*N

trials = 100
n_steps = int(T // dt)
alive_counts = np.zeros((n_steps, N))

avg_survival_rate = np.zeros((int(T//dt)))
all_times = []

alive_all_ind = np.zeros((n_steps, N, trials), dtype=bool)

S1_counts = np.zeros(n_steps)
S2_counts = np.zeros(n_steps)

for trial in range(trials):
    agents, alive = simulate(x0=x0,ideologies=ideologies, T=T, N=N, mu=mu, sigma=sigma, dt=dt)
    alive_counts += alive
    S1_counts += (alive[:, 0] & alive[:, 1])
    S2_counts += (alive[:, 0] | alive[:, 1])

S1_sim = S1_counts / trials
S2_sim = S2_counts / trials

    

survival_prob = alive_counts / trials

# plot
time = np.arange(n_steps) * dt
fig, ax = plt.subplots()
#ax.loglog(time[1:], survival_prob[1:, 0], label='S1')
#ax.loglog(time[1:], survival_prob[1:, 1], label='S2')
#ax.plot(time[1:], survival_prob[1:, 0], label='S1')
#ax.plot(time[1:], survival_prob[1:, 1], label='S2')
#ax.plot(time[1:], survival_prob[1:, 0], 'r--', label='Altruists (sim)')
ax.plot(time[1:], S1_sim[1:], 'b--', label='1st individualist (sim)')
ax.plot(time[1:], S2_sim[1:], 'g--', label='2nd individualist (sim)')

ax.set_xlabel('Time')
ax.set_ylabel('Survival Probability')
ax.set_ylim([0,1])
ax.set_xlim([0,10])

ax.legend()
ax.set_xlabel("t")
ax.set_ylabel("$S(x_0=1,t)$")
plt.show()
sys.exit()



"""fig, ax = plt.subplots(1,1,figsize=(4,3),dpi=300,tight_layout=True)
ax.plot(np.arange(int(T//dt)), avg_survival_rate)
plt.show()
sys.exit()
"""


fig, ax = plt.subplots(2, 1, figsize=(4, 3), dpi=300, tight_layout=True)
axes = [ax[i] for i in range(2)]
for i in range(N):
    axes[0].plot(
        np.arange(len(agents[i].x)),
        agents[i].x,
        color="r" if agents[i].bankrupt[-1] else "gray",
        alpha=0.5,
    )
axes[1].plot(np.arange(len(agents[i].x)), survival_rate[:len(agents[i].x)])
axes[1].set_ylim([0, 1])
axes[1].set_ylabel("Surival Rate")
axes[1].set_xlabel("Time")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("$x_i$")
sns.despine(ax=axes[0])
sns.despine(ax=axes[1])
plt.show()
