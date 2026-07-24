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
        self.x = [x0]
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.bankrupt_bound = lower_bound
        self.bankrupt = [False]
        self.ideology = ideology

    def dx(self):
        """
        Returns the SDE value for the drift diffusion model.
        Noise term is proportional to sqrt(dt)

        Parameters
        -----------
        mu : float
            drift rate or bias strength
        dt : float
            (optional) time step
        sigma : float
            (optional) diffusion or noise magnitude

        Returns
        -------
        float
            value of drift diffusion model
        """
        return (
            self.mu * self.dt
            + self.sigma * np.sqrt(self.dt) * self.rng.standard_normal()
        )

    # sets bankrupt to bankrupt_bound
    def update2(self):
        new_x = self.x[-1] + self.dx()
        if self.bankrupt[-1] or new_x <= self.bankrupt_bound:
            self.x.append(self.bankrupt_bound)
            self.bankrupt.append(True)
        else:
            self.x.append(new_x)
            self.bankrupt.append(False)

    # sets bankrupt to achieved value
    def update(self):
        if self.bankrupt[-1]:
            self.x.append(self.x[-1])  # already dead, frozen
            self.bankrupt.append(True)
            return
        new_x = self.x[-1] + self.dx()
        self.x.append(new_x)  # keep true value, even if slightly negative
        self.bankrupt.append(new_x <= self.bankrupt_bound)
