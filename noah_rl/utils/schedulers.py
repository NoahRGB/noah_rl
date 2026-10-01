from abc import ABC, abstractmethod

class Scheduler(ABC):
    @abstractmethod
    def step(self, n):
        ...

    @abstractmethod
    def get_value(self):
        ...

def detect_scheduler(scheduler) -> Scheduler|None:
    if scheduler.type == "linear":
        return LinearScheduler(**scheduler.params)
    else:
        return None

class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, steps):
        self.start_value = start_value
        self.current_value = start_value
        self.current_step = 0
        self.end_value = end_value
        self.decay_steps = steps

    def step(self, n=1):
        self.current_step = min(self.current_step + n, self.decay_steps)
        progress = self.current_step / self.decay_steps
        self.current_value = self.start_value + (self.end_value - self.start_value) * progress

        return self.current_value

    def get_value(self):
        return self.current_value

