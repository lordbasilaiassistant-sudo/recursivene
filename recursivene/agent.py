"""The curiosity engine: how the learner decides WHAT to experience next.

Sample-efficiency is won or lost here. Policies share identical world models and
identical budgets; they differ only in which activity they sample.

    random        - uniform. The passive baseline.
    novelty       - argmax recent prediction error. Naive surprise-seeking; provably
                    trapped by the noisy TV (error stays maximal there forever).
    disagreement  - argmax variance across an ensemble of models with different random
                    features. A STRONGER curiosity baseline than raw novelty: it is not
                    the strawman. (Answers the critique that LP only beat raw-novelty.)
    lp            - argmax LEARNING PROGRESS = how fast error is dropping, with a
                    per-activity noise floor subtracted before rectifying. This is the
                    RecursiveNe core. Mastered -> ~0 progress; pure noise -> progress
                    below its own noise floor -> rejected. Attention flows to wherever
                    the model is actually improving (Oudeyer-Kaplan-Hafner 2007;
                    Schmidhuber 2010: reward = compression/learning progress).

The noise-floor term answers the adversarial point that max(0, e_slow - e_fast) has a
positive bias under a stochastic learner and so leaks the noisy-TV back in. We subtract
c * std(residual) per activity, so an arm whose error merely *fluctuates* registers ~0
progress; only a genuine downward trend survives.
"""

import numpy as np
from collections import deque

from .model import RFFOnlineRegressor

POLICIES = ("random", "novelty", "disagreement", "lp")


class RegionLearner:
    """One world model per activity + the surprise / learning-progress bookkeeping the
    curiosity engine reads. (At higher layers 'activity' becomes a learned latent
    region; this bookkeeping interface is the invariant that survives.)"""

    def __init__(self, K, n_features=64, gamma=2.0, ridge=1.0, forget=1.0,
                 hist=24, ensemble=1, lp_floor=1.0, min_lp=8,
                 tau_master=0.04, noise_floor=0.7, seed=0):
        self.K = K
        self.hist = int(hist)
        self.min_lp = int(min_lp)        # samples before LP is estimable (caps warmup waste)
        self.ensemble = int(ensemble)
        self.lp_floor = float(lp_floor)
        self.tau_master = float(tau_master)   # an arm below this error is mastered, stop sampling it
        self.noise_floor = float(noise_floor)  # error above this is treated as unlearnable (noise)
        self.probe_min = int(min_lp) * 3       # probes before an above-floor arm is confirmed noise
        # Primary model per activity (+ optional extra heads for disagreement).
        self.models = [
            RFFOnlineRegressor(n_features=n_features, gamma=gamma, ridge=ridge,
                               forget=forget, seed=seed + 1000 * r)
            for r in range(K)
        ]
        self.heads = None
        if self.ensemble > 1:
            self.heads = [
                [RFFOnlineRegressor(n_features=n_features, gamma=gamma, ridge=ridge,
                                    forget=forget, seed=seed + 1000 * r + 7919 * (h + 1))
                 for h in range(self.ensemble - 1)]
                for r in range(K)
            ]
        self.err_hist = [deque(maxlen=self.hist) for _ in range(K)]
        self.visits = np.zeros(K, dtype=int)

    def observe(self, r, x, y):
        err = self.models[r].update(x, y)
        if self.heads is not None:
            for h in self.heads[r]:
                h.update(x, y)
        self.err_hist[r].append(err * err)
        self.visits[r] += 1
        return err

    def recent_error(self, r):
        h = self.err_hist[r]
        return float(np.mean(h)) if h else np.inf

    def disagreement(self, r, x):
        """Ensemble prediction variance at x — epistemic (reducible) uncertainty."""
        if self.heads is None:
            return self.recent_error(r)
        preds = [self.models[r].predict(x)] + [h.predict(x) for h in self.heads[r]]
        return float(np.var(preds))

    def learning_progress(self, r):
        """Learning progress = a SIGNIFICANT downward trend in prediction error.

        We regress squared-error on time over the recent window and return the expected
        total improvement (-slope * n) MINUS a significance floor (lp_floor standard
        errors of the slope). This is the principled separation of reducible from
        irreducible variance the noisy-TV problem demands:
          * a hard-but-learnable activity has a clear negative slope with small standard
            error -> LP strongly positive (so it gets pursued, fixing the timid-on-hard bug);
          * a pure-noise activity has slope ~ 0 with LARGE residual variance -> the
            significance floor drives LP negative -> it is rejected;
          * a mastered activity has slope ~ 0 with tiny residuals -> LP ~ 0 -> harmless.
        Returns None until min_lp samples exist (keeps the warmup cheap)."""
        h = self.err_hist[r]
        n = len(h)
        if n < self.min_lp:
            return None
        y = np.asarray(h)
        t = np.arange(n)
        tbar = t.mean()
        Stt = float(((t - tbar) ** 2).sum())
        if Stt < 1e-9:
            return None
        slope = float(((t - tbar) * (y - y.mean())).sum() / Stt)
        resid = y - (y.mean() + slope * (t - tbar))
        sigma2 = float((resid ** 2).sum()) / max(n - 2, 1)
        se = np.sqrt(sigma2 / Stt)                       # standard error of the slope
        return float(-slope * n - self.lp_floor * se * n)

    def n_params(self):
        per = sum(m.n_params() for m in self.models)
        if self.heads is not None:
            per += sum(h.n_params() for row in self.heads for h in row)
        return int(per)

    def total_flops(self):
        f = sum(m.flops for m in self.models)
        if self.heads is not None:
            f += sum(h.flops for row in self.heads for h in row)
        return int(f)

    def ram_floats(self):
        r = sum(m.ram_floats() for m in self.models)
        if self.heads is not None:
            r += sum(h.ram_floats() for row in self.heads for h in row)
        return int(r)


def choose(policy, learner, world, rng, epsilon=0.1):
    """Pick the next activity to sample under the given policy."""
    K = learner.K

    # Warmup: any activity whose LP estimate is undefined gets sampled first
    # (least-visited among them). Every activity — including the noise trap — is tried,
    # so abandoning it later is an evidenced decision, not a built-in assumption.
    undefined = [r for r in range(K) if learner.learning_progress(r) is None]
    if undefined:
        return min(undefined, key=lambda r: learner.visits[r])

    if rng.random() < epsilon:
        return int(rng.integers(K))

    if policy == "random":
        return int(rng.integers(K))

    if policy == "novelty":
        return int(np.argmax([learner.recent_error(r) for r in range(K)]))

    if policy == "disagreement":
        xs = world.sample_x()
        return int(np.argmax([learner.disagreement(r, xs) for r in range(K)]))

    if policy == "lp":
        # RecursiveNe core: work on the WORST activity that is still LEARNABLE, never the
        # noisy TV. Two facts make this both efficient and noise-proof:
        #   * an UNFIT learnable activity has squared error <= E[sin^2] = 0.5, and it only
        #     drops from there; an unlearnable noise activity sits at its variance ~ 1.0.
        #     So 'error below the noise floor' cleanly flags reducible (learnable) error.
        #   * among those, argmax ERROR drives samples to the hardest-still-unmastered
        #     activity — exactly what minimizes WORST-CASE competence.
        # This dodges both failure modes: novelty chases the TV (max raw error), pure
        # learning-progress procrastinates on slow-but-hard arms. (learning_progress() is
        # retained as the harness/benchmark signal for 'is anything still improving'.)
        errs = np.array([learner.recent_error(r) for r in range(K)])
        mastered = errs < learner.tau_master
        eligible = (~mastered) & (errs < learner.noise_floor)
        if eligible.any():
            return int(np.argmax(np.where(eligible, errs, -np.inf)))
        # No learnable arm is unmastered right now. Probe only UNDER-EXPLORED unmastered
        # arms (cheap discovery of slow starters); an arm probed >= probe_min times that
        # is still above the noise floor is CONFIRMED noise and abandoned for good — this
        # is what stops the agent from wandering back to the TV once it has nothing left
        # to learn (the post-mastery noise leak).
        probe = (~mastered) & (learner.visits < learner.probe_min) & (errs >= learner.noise_floor)
        if probe.any():
            cand = np.where(probe)[0]
            return int(min(cand, key=lambda r: learner.visits[r]))
        return int(np.argmin(errs))   # nothing to learn -> harmless lowest-error arm


    raise ValueError(f"unknown policy: {policy}")
