import numpy as np
# import warnings
from abc import ABC
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class TimeGeneratorBase(ABC):
    def __init__(self, start_time=0.0):
        self.start_time = start_time
        self.cur_time = self.start_time
        self.prev_time = self.start_time

    def get_cur_time(self):
        return self.cur_time

    def __iter__(self):
        return self


class UniformTimeGenerator(TimeGeneratorBase):
    def __init__(self, start_time=0.0, time_step=1.0):
        super().__init__(start_time)
        self.time_step = time_step

    def __next__(self):
        # self.prev_time = self.cur_time
        self.cur_time += self.time_step
        return self.cur_time


class StochasticTimeGenerator(TimeGeneratorBase):
    def __init__(self, start_time=0.0, mean=1.0, deviation=1.0):
        super().__init__(start_time)
        self.random_generator = np.random.default_rng()
        self.delta_mean = mean
        self.delta_deviation = deviation

    def __next__(self):
        # self.prev_time = self.cur_time
        self.cur_time += self.random_generator.normal(loc=self.delta_mean, scale=self.delta_deviation)
        return self.cur_time


class TrajectoryGeneratorBase(ABC):
    def __init__(self, time_generator, start_pt=np.zeros(3).T):
        self.time_generator = time_generator
        self.ndims = start_pt.shape[0]
        # self.cur_vec = np.zeros(self.ndims).T
        # self.time_generator = TimeGeneratorBase()
        self.cur_vec = start_pt
        self.cur_time = self.time_generator.get_cur_time()
        self.prev_time = self.cur_time

    def get_ndims(self):
        return self.ndims

    def set_timegenerator(self, tg):
        self.time_generator = tg

    def __next__(self):
        self.prev_time = self.cur_time
        self.cur_time = next(self.time_generator)
        delta_t = self.cur_time - self.prev_time
        return self.cur_time, self.gen_next_pt(delta_t)

    def __iter__(self):
        return self


class TrajectoryGeneratorLinear(TrajectoryGeneratorBase):
    def __init__(self, evo_mat):
        super().__init__()
        self.evo_mat = evo_mat

    def gen_next_pt(self, delta_t):
        return self.cur_vec * self.evo_mat * delta_t


class TrajectoryGeneratorHarmonic(TrajectoryGeneratorBase):
    def __init__(self,
                 time_generator,
                 center_pos=np.zeros(2).T,
                 a_arr=np.ones(2).T,
                 omega_arr=2.0 * np.pi * np.ones(2).T,
                 phase_arr=np.zeros(2).T):
        start_pt = a_arr * np.sin(omega_arr * time_generator.start_time + phase_arr)
        super().__init__(time_generator=time_generator, start_pt=start_pt)
        self.phase_arr = phase_arr
        self.omega_arr = omega_arr
        self.a_arr = a_arr

        self.cur_phase = self.phase_arr

    def to_descartes(self):
        return self.a_arr * np.sin(self.phase_arr)

    def gen_next_pt(self, delta_t):
        self.phase_arr += self.omega_arr * delta_t
        return self.to_descartes()


if __name__ == "__main__":
    timegenclass = StochasticTimeGenerator(start_time=0.0, mean=1.0, deviation=0.1)
    trajectoryclass = TrajectoryGeneratorHarmonic(time_generator=timegenclass, omega_arr=np.array([9.0, 10.0]))

    x = []
    y = []
    for i in range(1000):
        cur_coord = (next(trajectoryclass)[1])
        x.append(cur_coord[0])
        y.append(cur_coord[1])

    plt.scatter(x, y)
    plt.show()


