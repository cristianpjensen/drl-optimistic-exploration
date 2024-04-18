from typing import List, Tuple

class LinearScheduler():
    """Linear interpolation between points."""

    def __init__(self, points: List[Tuple[int, float]]):
        self.points = points

    def value(self, t):
        for i in range(len(self.points) - 1):
            start_t, start_v = self.points[i]
            end_t, end_v = self.points[i + 1]

            if t >= start_t and t < end_t:
                fraction = (t - start_t) / (end_t - start_t)
                return start_v + fraction * (end_v - start_v)

        return self.points[-1][1]


if __name__ == "__main__":
    # Testing
    scheduler = LinearScheduler([(0, 1), (10, 0.1), (100, 0.01)])

    for t in range(0, 101):
        print(f"{t}, {scheduler.value(t)}")
