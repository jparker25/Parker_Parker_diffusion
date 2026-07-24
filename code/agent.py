"""
agent.py

Class for modeling an agent going through a DDM model.

Authors: John E. Parker (2026)
"""

# import python libraries
import numpy as np


class Agent:
    def __init__(
        self,
        x0=0.5,
        mu=0,
        dt=0.001,
        sigma=0.3,
        lower_bound=0,
        ideology="individualist",
        seed=1,
    ):
        self.x = [x0]  # initial value for wealth / resource
        self.mu = mu  # drift term
        self.dt = dt  # time step
        self.sigma = sigma  # noise volatility
        self.seed = seed  # random seed
        self.rng = np.random.default_rng(self.seed)  # set RNG based on random seed
        self.bankrupt_bound = lower_bound  # bound for 'death' or 'bankruptcy'
        self.bankrupt = [False]  # Start as no death
        self.ideology = ideology  # Set ideology

    def dx(self):
        """
        Returns the SDE value for the drift diffusion model.
        Noise term is proportional to sqrt(dt)

        Parameters
        ----------
        self
            Agent class

        Returns
        -------
        float
            value of drift diffusion model
        """
        return (
            self.mu * self.dt
            + self.sigma * np.sqrt(self.dt) * self.rng.standard_normal()
        )

    def update2(self):
        """
        Updates the values for bankruptcy (T/F) and current wealth.
        Lower bound of x is bankruptcy value.
        """
        # update values
        new_x = self.x[-1] + self.dx()

        # check if bankrupt or below bankruptcy bound, otherwise update values
        if self.bankrupt[-1] or new_x <= self.bankrupt_bound:
            self.x.append(self.bankrupt_bound)
            self.bankrupt.append(True)
        else:
            self.x.append(new_x)
            self.bankrupt.append(False)

    # sets bankrupt to achieved value
    def update(self):
        """
        Updates the values for bankruptcy (T/F) and current wealth.
        Lower bound of x is current value (can be below bankruptcy).
        """

        # check if bankrupt
        if self.bankrupt[-1]:
            self.x.append(self.x[-1])  # already dead, frozen
            self.bankrupt.append(True)
            return

        # update values otherwise
        new_x = self.x[-1] + self.dx()
        self.x.append(new_x)  # keep true value, even if slightly negative
        self.bankrupt.append(new_x <= self.bankrupt_bound)
