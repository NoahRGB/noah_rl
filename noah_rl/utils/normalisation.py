# NoahRGB 09/2026
#
# a file that contains some functionality for keeping track of a
# running mean/stdev of some values over time and normalising
# using them
#

import numpy as np

class Normaliser:
    # see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    def __init__(self, dim, use_mean=True):
        self.dim = dim
        self.use_mean = use_mean
        self.count = 1e-4 # prevent divide by 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.ones(dim, dtype=np.float64)

    def add_batch(self, batch: np.ndarray):
        # expecting a numpy array of shape
        # (batch, self.dim)

        # M2 = sum of squared deviations

        batch_size = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)

        batch_mean = batch_mean.reshape(self.dim)
        batch_var = batch_var.reshape(self.dim)

        batch_m2 = batch_var * batch_size

        assert batch_mean.shape == self.mean.shape

        new_count = self.count + batch_size
        mean_delta = self.mean - batch_mean
        self.mean = (batch_size * batch_mean + self.count * self.mean) / new_count
        self.m2 = batch_m2 + self.m2 + mean_delta**2 * batch_size * self.count / new_count
        self.count = new_count

    def normalise(self, inp: np.ndarray, clip_range: list = None):
        var = self.m2 / self.count
        ep = 1e-8

        if self.use_mean:
            normalised = (inp - self.mean) / np.sqrt(var + ep)
        else:
            normalised = inp / np.sqrt(var + ep)

        if clip_range is not None:
            return np.clip(normalised, clip_range[0], clip_range[1])
        
        return normalised

    def save(self):
        return self.count, self.mean, self.m2

    def load(self, data: tuple):
        self.count, self.mean, self.m2 = data
