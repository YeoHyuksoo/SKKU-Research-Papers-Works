import numpy as np
from sklearn.neighbors import NearestNeighbors

from som_anomaly_detector.kohonen_som import KohonenSom

class AnomalyDetection(KohonenSom):
    def __init__(
        self,
        shape,
        input_size,
        learning_rate,
        learning_decay=0.1,
        initial_radius=1,
        radius_decay=0.1,
        min_number_per_bmu=1,
        number_of_neighbors=3,
    ):
        super(AnomalyDetection, self).__init__(
            shape,
            input_size,
            learning_rate,
            learning_decay,
            initial_radius,
            radius_decay,
        )

        self.minNumberPerBmu = min_number_per_bmu
        self.numberOfNeighbors = number_of_neighbors
        return

    def get_bmu_counts(self, training_data):
        bmu_counts = np.zeros(shape=self.shape)
        for observation in training_data:
            bmu = self.get_bmu(observation)
            bmu_counts[bmu] += 1
        return bmu_counts

    def fit(self, training_data, num_iterations):
        self.reset()
        super(AnomalyDetection, self).fit(training_data, num_iterations)
        bmu_counts = self.get_bmu_counts(training_data)
        self.bmu_counts = bmu_counts.flatten()
        self.allowed_nodes = self.grid[bmu_counts >= self.minNumberPerBmu]
        return self.allowed_nodes

    def evaluate(self, evaluationData):
        self.allowed_nodes
        assert self.allowed_nodes.shape[0] > 1
        else:
            classifier = NearestNeighbors(n_neighbors=self.numberOfNeighbors)
            classifier.fit(self.allowed_nodes)
            dist, _ = classifier.kneighbors(evaluationData)
        return dist.mean(axis=1)
