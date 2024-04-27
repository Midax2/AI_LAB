import torch
from torch import nn, optim
import torch.nn.functional as F
from data import ClassificationTwoSpiralsData
from visualization_utils import inspect_data, plot_data, x_data_from_grid
import matplotlib.pyplot as plt


class TorchMultiLayerNetwork(nn.Module):
    """
    Prosta sieć neuronowa w pyTorch o parametryzowalnej liczbie warstw ukrytych.
    Konstrukcja analogiczna, jak dwuwarstwowej sieci z części pierwszej.

    Np. dla n_in=2, n_hiddens=[10, 20] i n_out=1 powstanie sieć z dwoma wejściami, pierwszą warstwą ukrytą z 10
    neuronami, drugą warstwą ukrytą z 20 neuronami i jednym neuronem w warstwie wyjściowej. Funkcje aktywacji
    ustawione "na sztywno": 'relu' w warstwach ukrytych i 'sigmoid' w warstwie wyjściowej.
    """

    def __init__(self, n_in, n_hiddens, n_out):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(n_i, n_o) for n_i, n_o in zip([n_in] + n_hiddens, n_hiddens + [n_out])])

    def forward(self, x_data):
        x = x_data
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return F.sigmoid(self.layers[-1](x))


def evaluate_model(model, x, y):
    with torch.no_grad():                       # przy ewaluacji nie ma potrzeby liczyć gradientów
        y_pred = torch.round(model.forward(x))
        acc = (y == y_pred).sum().item() / y.shape[0]
    return acc


def training(model, x, y):
    # training hyperparameters
    n_steps = 50000
    learning_rate = 0.1
    minibatch_size = 32

    loss_fn = F.binary_cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # SGD optimizer

    history = []
    i_step = 0

    for _ in range(n_steps):
        # Random mini-batch data selection
        indices = torch.randint(0, x.size(0), (minibatch_size,))
        x_batch, y_batch = x[indices], y[indices]

        # Forward pass and loss calculation
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        # Track stats
        history.append(loss.item())
        i_step += 1
        if i_step % 1000 == 0:
            print(f'Step={i_step}, lr={learning_rate:.6f}, loss={loss.item():.6f}')

    return model, history


def plot_history(history, smoothing_factor=0.99):
    plt.plot(history, c='b', alpha=0.5)
    if smoothing_factor is not None:
        smoothed_history = []; pv = history[0]
        for v in history:
            smoothed_history.append(smoothed := smoothing_factor*pv + (1.0-smoothing_factor)*v)
            pv = smoothed
        plt.plot(smoothed_history, c='b')
    plt.xlabel('Training step')
    plt.ylabel('Log loss')
    plt.show()


def classify_spirals(student_id, do_data_inspection=True, do_model_inpection=True, do_model_training=True,
                     load_trained_model=False):

    model_path = "my_model.pkl"

    # generacja danych
    n_examples = 1000
    gen = ClassificationTwoSpiralsData(degrees=900)
    x, y = gen.generate_data(n_examples=n_examples, noise=3.0, seed=student_id)

    # podział na zbiór uczący i testowy:
    n_train = int(0.8 * n_examples)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    if do_data_inspection:
        inspect_data(x_train, y_train, x_test=x_test, y_test=y_test)
        plot_data(x, y, x_test=x_test, y_test=y_test, plot_xy_range=[-1.2, 1.2])

    # konwersja danych do torch.Tensor
    x_train, y_train, x_test, y_test = [torch.from_numpy(x) for x in [x_train, y_train, x_test, y_test]]

    # utworzenie modelu
    model = TorchMultiLayerNetwork(
        n_in=2,
        n_hiddens=[16, 32, 32, 16],
        n_out=1)

    if load_trained_model:
        model.load_state_dict(torch.load(model_path))
        print(f"Wagi modelu załadowane z {model_path}")

    if do_model_inpection:
        print(model)
        print("Aktualne wagi:")
        for name, param in model.named_parameters():
            print(name, param.data)                 # początkowo wagi losowe
        print("Aktualne gradienty:")
        for name, param in model.named_parameters():
            print(name, param.requires_grad)        # czy dla tego parametru będzie potrzebne liczenie gradientu
            print(name, param.grad)                 # aktualna wartość gradientu

    print(f'Train accuracy: {evaluate_model(model, x_train, y_train) * 100:.02f}%')
    print(f'Test accuracy: {evaluate_model(model, x_test, y_test) * 100:.02f}%')

    if do_model_training:
        model, history = training(model,
                                  x_train,
                                  y_train)
        plot_history(history)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}.")

    # ewaluacja po treningu
    print(f'Train accuracy: {evaluate_model(model, x_train, y_train) * 100:.02f}%')
    print(f'Test accuracy: {evaluate_model(model, x_test, y_test) * 100:.02f}%')

    # test na całej przestrzeni wejść (z wizualizacją)
    x_grid = x_data_from_grid(min_xy=-1.2, max_xy=1.2, grid_size=1000)
    with torch.no_grad():
        y_pred_grid = model.forward(torch.from_numpy(x_grid))
    plot_data(x_train, y_train, x_test=x_test, y_test=y_test,
              plot_xy_range=[-1.2, 1.2], x_grid=x_grid, y_grid=y_pred_grid.numpy())


if __name__ == '__main__':

    student_id = 196751         # Twój numer indeksu, np. 102247
    torch.manual_seed(student_id)

    classify_spirals(student_id,
                     do_data_inspection=True,
                     do_model_inpection=True,
                     do_model_training=True,
                     load_trained_model=False)
