import numpy as np
from data import LinearlySeparableClasses, NonlinearlySeparableClasses
from visualization_utils import inspect_data, plot_data, x_data_from_grid, visualize_activation_function, \
    plot_two_layer_activations


# Przykładowe funkcje aktywacji

def relu(logits):
    return np.maximum(logits, 0)

def sigmoid(logits):
    return 1. / (1. + np.exp(-logits))
    # return np.exp(-np.logaddexp(0, -logits))     # to samo co wyżej, ale stabilne numerycznie

def hardlim(logits):
    return (logits > 0).astype(np.float32)
    # return np.round(sigmoid(logits))             # to samo co wyżej, bez wykorzystywania porównań i rzutowań

def linear(logits):
    return logits


def zad1_single_neuron(student_id):
    gen = LinearlySeparableClasses()
    x, y = gen.generate_data(seed=student_id)
    n_samples, n_features = x.shape

    # zakomentuj, jak juz nie potrzebujesz
    inspect_data(x, y)
    plot_data(x, y, plot_xy_range=[-1, 2])

    # model pojedynczego neuronu
    class SingleNeuron:
        def __init__(self, n_in, f_act):
            self.W = 0.01 * np.random.randn(n_in, 1)  # rozmiar W: [n_in, 1]
            self.b = 0.01 * np.random.randn(1)  # rozmiar b: [1]
            self.f_act = f_act

        def forward(self, x_data):
            """
            :param x_data: wejście neuronu: np.array o rozmiarze [n_samples, n_in]
            :return: wyjście neuronu: np.array o rozmiarze [n_samples, 1]
            """
            # TODO (0.5 point)
            logits = np.dot(x_data, self.W) + self.b
            return self.f_act(logits)

    # neuron zainicjowany losowymi wagami
    model = SingleNeuron(n_in=n_features, f_act=hardlim)

    # TODO: ustawienie właściwych wag (0.5 point)
    # model.W[:, 0] = [w1, w2]
    # model.b[:] = [b1]

    # działanie i ocena modelu
    y_pred = model.forward(x)
    print(f'Accuracy = {np.mean(y == y_pred) * 100}%')

    # test na całej przestrzeni wejść (z wizualizacją)
    x_grid = x_data_from_grid(min_xy=-1, max_xy=2, grid_size=1000)
    y_pred_grid = model.forward(x_grid)
    plot_data(x, y, plot_xy_range=[-1, 2], x_grid=x_grid, y_grid=y_pred_grid, title='Linia decyzyjna neuronu')


def zad2_two_layer_net(student_id):
    gen = NonlinearlySeparableClasses()
    x, y = gen.generate_data(seed=student_id)
    n_samples, n_features = x.shape

    # zakomentuj, jak juz nie potrzebujesz
    inspect_data(x, y)
    plot_data(x, y, plot_xy_range=[-1, 2])

    # warstwa czyli n_out pojedynczych, niezależnych neuronów operujących na tym samym wejściu\
    # (i-ty neuron ma swoje parametry w i-tej kolumnie macierzy W i na i-tej pozycji wektora b)
    class DenseLayer:
        def __init__(self, n_in, n_out, f_act):
            self.W = 0.01 * np.random.randn(n_in, n_out)  # rozmiar W: ([n_in, n_out])
            self.b = 0.01 * np.random.randn(n_out)  # rozmiar b  ([n_out])
            self.f_act = f_act

        def forward(self, x_data):
            logits = np.dot(x_data, self.W) + self.b
            return self.f_act(logits)

    class SimpleTwoLayerNetwork:
        def __init__(self, n_in, n_hidden, n_out):
            self.hidden_layer = DenseLayer(n_in, n_hidden, relu)
            self.output_layer = DenseLayer(n_hidden, n_out, sigmoid)

        def forward(self, x_data):
            hidden_output = self.hidden_layer.forward(x_data)
            return self.output_layer.forward(hidden_output)

    # model zainicjowany losowymi wagami
    model = SimpleTwoLayerNetwork(n_in=n_features, n_hidden=2, n_out=1)

    # TODO: ustawienie właściwych wag
    model = SimpleTwoLayerNetwork(n_in=n_features, n_hidden=2, n_out=1)
    model.hidden_layer.W[:, 0] = [w11, w21]  # wagi dla neuronu h1
    model.hidden_layer.W[:, 1] = [w12, w22]  # wagi dla neuronu h2
    model.hidden_layer.b[:] = [b1, b2]  # biasy dla neuronów h1 i h2
    model.output_layer.W[:, 0] = [w_out1]  # wagi dla neuronu wyjściowego
    model.output_layer.b[:] = [b_out]  # bias dla neuronu wyjściowego

    # działanie i ocena modelu
    y_pred = model.forward(x)
    print(f'Accuracy = {np.mean(y == y_pred) * 100}%')

    plot_two_layer_activations(model, x, y)


if __name__ == '__main__':
    # visualize_activation_function(relu)

    student_id = None         # Twój numer indeksu, np. 102247

    zad1_single_neuron(student_id)
    # zad2_two_layer_net(student_id)
