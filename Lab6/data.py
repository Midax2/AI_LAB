import numpy as np
import math
from abc import ABC, abstractmethod


class DataGenerator(ABC):
    @abstractmethod
    def generate_data(self, n_examples: int, *args, **kwargs):
        pass


class ClassificationTwoSpiralsData(DataGenerator):
    def __init__(self, degrees=570, start=40, min_x=-12, max_x=12):
        """
        Adopted from Matlab code at:
            http://www.mathworks.com/matlabcentral/fileexchange/
            41459-6-functions-for-generating-artificial-datasets/content/twospirals.m

        :type degrees: int
        :param degrees: length of the spirals

        :type start: int
        :param start: how far from the origin the spirals start, in degrees

        :param min_x:
        :param max_x:
        """
        super().__init__()
        self.degrees = degrees
        self.start = start
        self.min_x = min_x
        self.max_x = max_x

    def generate_data(self, n_examples=2000, noise=0.5, seed=1234, **kwargs):
        """
        :type n_examples: int
        :param n_examples: number of instances
        :type noise: float
        :param noise: 0 is no noise, at 1 the spirals will start to overlap
        :type seed: int
        :param seed:
        """

        deg2rad = 2. * math.pi / 360.
        start = self.start * deg2rad

        N1 = int(math.floor(n_examples / 2))
        N2 = n_examples - N1

        rng = np.random.RandomState(seed)

        n = start + np.sqrt(rng.rand(N1)) * self.degrees * deg2rad
        d1 = np.column_stack((-np.cos(n) * n + rng.rand(N1) * noise,
                              np.sin(n) * n + rng.rand(N1) * noise, np.zeros_like(n)))

        n = start + np.sqrt(rng.rand(N2)) * self.degrees * deg2rad
        d2 = np.column_stack((np.cos(n) * n + rng.rand(N2) * noise,
                              -np.sin(n) * n + rng.rand(N2) * noise, np.ones_like(n)))

        d = np.vstack((d1, d2))

        order = np.arange(len(d))
        rng = np.random.RandomState(seed)
        rng.shuffle(order)
        d = d[order]
        data_x = d[:, 0:2].astype(np.float32)
        data_labels = d[:, 2]
        data_labels = data_labels.reshape(data_labels.size, 1).astype(np.float32)

        min_data_x = np.min(data_x)
        max_data_x = np.max(data_x)
        data_x = (data_x - min_data_x)/(max_data_x - min_data_x) * 2. - 1.

        return data_x, data_labels


class LogicGateData(DataGenerator):
    def __init__(self, gate_type='xor', dtype=np.float32):
        super().__init__()
        supported_gate_types = ['or', 'nor', 'xor', 'and', 'nand']
        if gate_type not in supported_gate_types:
            raise ValueError(f'Not supported gate type {gate_type}. Choose one of: {supported_gate_types}')
        self._gate_type = gate_type
        self.dtype = dtype

    def generate_data(self, **kwargs):
        x_data = np.array([[0, 0],
                           [0,  1],
                           [ 1, 0],
                           [ 1,  1]], dtype=np.float32)
        if self._gate_type == 'xor':
            labels = np.logical_xor(x_data[:, 0], x_data[:, 1])
        elif self._gate_type == 'and':
            labels = np.logical_and(x_data[:, 0], x_data[:, 1])
        elif self._gate_type == 'nand':
            labels = 1 - np.logical_and(x_data[:, 0], x_data[:, 1])
        elif self._gate_type == 'nor':
            labels = 1 - np.logical_or(x_data[:, 0], x_data[:, 1])
        else:
            labels = np.logical_or(x_data[:, 0], x_data[:, 1])

        return x_data, labels.astype(self.dtype).reshape((4, 1))


class RotatedLogicGateData(LogicGateData):
    def __init__(self, gate_type='xor',  dtype=np.float32):
        super().__init__(gate_type=gate_type, dtype=dtype)

    def generate_data(self, seed, **kwargs):
        x_data, labels = super().generate_data()
        if seed is not None:
            np.random.seed(seed)
        x_data, labels = np.tile(x_data, (10, 1)), np.tile(labels, (10, 1))
        x_data += 0.2*np.random.rand(x_data.shape[0], 2) - 0.1
        theta = np.radians(np.random.randint(360))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])

        return (x_data - 0.5) @ R + 0.5, labels


class LinearlySeparableClasses(RotatedLogicGateData):
    def __init__(self):
        super().__init__(gate_type='and')


class NonlinearlySeparableClasses(RotatedLogicGateData):
    def __init__(self):
        super().__init__(gate_type='xor')


def input_noise(datax, noise_level=0.01, dtype=np.float32):
    noisy_datax = datax + noise_level * np.random.randn(datax.size).reshape(datax.shape).astype(dtype)
    return noisy_datax


def label_noise(labels, noise_level=0.05, dtype=np.float32):
    r = np.random.rand(labels.size).reshape(labels.shape) < noise_level
    noisy_labels = np.logical_xor(labels, r).astype(dtype)
    return noisy_labels
