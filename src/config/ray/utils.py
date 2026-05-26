from ray.tune.stopper import Stopper

class PatienceStopper(Stopper):
    def __init__(self, metric="test_loss", patience=8, mode="min"):
        self.metric = metric
        self.patience = patience
        self.mode = mode

        # Track per-trial state
        self.best = {}
        self.last_improved = {}

    def __call__(self, trial_id, result):
        value = result.get(self.metric)
        if value is None:
            return False  # cannot judge yet

        # Initialize dict entries
        if trial_id not in self.best:
            self.best[trial_id] = value
            self.last_improved[trial_id] = 0
            return False

        # Check improvement
        improved = (
            value < self.best[trial_id]
            if self.mode == "min"
            else value > self.best[trial_id]
        )

        if improved:
            self.best[trial_id] = value
            self.last_improved[trial_id] = 0
        else:
            self.last_improved[trial_id] += 1

        # Stop if patience exceeded
        return self.last_improved[trial_id] >= self.patience

    def stop_all(self):
        return False

