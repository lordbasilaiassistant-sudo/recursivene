"""The world-model primitive: an online, closed-form, backprop-free predictor, with a
FLOP meter so 'cost-for-competence' is a real compute measurement (not a proxy).

Random Fourier Features (Rahimi & Recht 2007) lift the scalar knob x into a fixed
nonlinear feature space; Recursive Least Squares (Plackett 1950) fits a linear map in
that space EXACTLY, one sample at a time, in closed form.

Why this embodies 'least training intensive' (A2): one observation -> exactly one
O(D^2) update. No epochs, no replay buffer, no minibatches, no global backprop. The
*prior* prediction error returned by update() IS the surprise the curiosity engine
reads — learning signal and intrinsic reward are one quantity computed once.

D (n_features) is the model's whole parameter budget and dominates the per-step FLOPs.
The RSI harness drives D down while holding competence: the 'race to 0' knob, made of
real arithmetic the meter counts.
"""

import numpy as np


class RFFOnlineRegressor:
    def __init__(self, n_features=64, gamma=2.0, ridge=1.0, forget=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.D = int(n_features)
        self.W = rng.normal(0.0, gamma, size=self.D)   # fixed random frequencies
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.D)
        self.scale = np.sqrt(2.0 / self.D)
        self.w = np.zeros(self.D)
        self.P = np.eye(self.D) / ridge
        self.forget = float(forget)
        self.n_updates = 0
        self.flops = 0          # cumulative FLOP meter

    def phi(self, x):
        return self.scale * np.cos(self.W * x + self.b)

    def predict(self, x):
        self.flops += 2 * self.D            # dot product
        return float(self.w @ self.phi(x))

    def update(self, x, y):
        """Fit (x, y); return the PRIOR prediction error (the surprise)."""
        p = self.phi(x)
        pred = float(self.w @ p)
        err = y - pred                      # surprise: error before this update
        Pp = self.P @ p                     # D^2
        denom = self.forget + float(p @ Pp)
        k = Pp / denom
        self.w = self.w + k * err
        self.P = (self.P - np.outer(k, Pp)) / self.forget   # D^2 outer + D^2 sub
        self.n_updates += 1
        self.flops += 4 * self.D * self.D + 4 * self.D       # RLS step ~ O(D^2)
        return err

    def n_params(self):
        return self.D

    def ram_floats(self):
        """Working memory in floats: weights (D) + inverse-covariance P (D*D)."""
        return self.D + self.D * self.D
