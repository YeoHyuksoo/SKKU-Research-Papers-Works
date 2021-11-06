import math
import numpy as np
from scipy.stats import multivariate_normal

# SOM Anomaly Detection
# This Python module provides implementaion of Kohonen's Self-Organizing-Maps for anomaly detection purposes.

# Simple description of the algorithm
# 1. Train a self organizing map of some dimension on the set of normal data.
# 2. Per node in the SOM, count the number of training vectors mapped to this node.
# 3. Remove all nodes with node which map training vectors less than a certain threshold.
# 4. For each observation in the data to be assessed, perform kNN wrt. And calculate the mean distance to the nodes found. This is the anomaly metric.
# 5. Order the evaluation data wrt. to the anomaly metric.

class KohonenSom(object):
    def __init__(
        self,
        shape,
        input_size,
        learning_rate,
        learning_decay=1,
        initial_radius=1,
        radius_decay=1,
    ):
        self.shape = shape
        self.dimension = shape.__len__()
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.initial_radius = initial_radius
        self.radius_decay = radius_decay

        distance = np.fromfunction(
            self._distance_function, tuple(2 * i + 1 for i in shape)
        )

        gaussian_transorm = np.vectorize(
            lambda x: multivariate_normal.pdf(x, mean=0, cov=1)
        )
        self.distance = gaussian_transorm(distance)

        self.distance = np.repeat(self.distance, self.input_size, self.dimension - 1)
        self.distance = np.reshape(
            self.distance, newshape=(distance.shape + (self.input_size,))
        )

        self.grid = np.random.rand(*(self.shape + (self.input_size,))) * 2 - 1
        return

    def reset(self):
        self.grid = np.random.rand(*(self.shape + (self.input_size,))) * 2 - 1
        return

    def _distance_function(self, *args):
        return sum([(i - x) ** 2 for i, x in zip(args, self.shape)])


    def get_bmu(self, sample):
        distances = np.square(self.grid - sample).sum(axis=self.dimension)
        bmu_index = np.unravel_index(distances.argmin().astype(int), self.shape)
        return bmu_index

    def fit(self, training_sample, num_iterations):
        sigma = self.initial_radius
        learning_rate = self.learning_rate
        for i in range(1, num_iterations):
            obs = training_sample[np.random.choice(training_sample.shape[0], 1)][0]
            bmu = self.get_bmu(obs)
            self.update_weights(obs, bmu, sigma, learning_rate)
            sigma = self.initial_radius * math.exp(-(i * self.radius_decay))
            learning_rate = self.learning_rate * math.exp(-(i * self.learning_decay))

        return self

    def update_weights(self, training_vector, bmu, sigma, learning_speed):
        reshaped_array = self.grid.reshape((np.product(self.shape), self.input_size))
        bmu_distance = self.distance
        for i, bmu_ind in enumerate(bmu):
            bmu_distance = np.roll(bmu_distance, bmu_ind, axis=i)

        for i, shape_ind in enumerate(self.shape):
            slc = [slice(None)] * len(bmu_distance.shape)
            slc[i] = slice(shape_ind, 2 * shape_ind)
            bmu_distance = bmu_distance[slc]

        bmu_distance = sigma * bmu_distance
        learning_matrix = -(self.grid - training_vector)
        scaled_learning_matrix = learning_speed * (bmu_distance * learning_matrix)
        self.grid = self.grid + scaled_learning_matrix

        return
