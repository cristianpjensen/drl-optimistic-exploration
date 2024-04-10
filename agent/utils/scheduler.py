class LinearScheduler():
    """Linear interpolation between `start` and `end` over `T` timesteps, after which `end` is
    returned.

    Args:
        T (int): The total number of timesteps for the interpolation.
        start (float): The starting value of the interpolation.
        end (float): The ending value of the interpolation.

    """

    def __init__(self, T: int, start: float, end: float):
        self.timesteps = T
        self.start = start
        self.end = end

    def value(self, t):
        fraction = min(t / self.timesteps, 1.0)
        return self.start + fraction * (self.end - self.start)
