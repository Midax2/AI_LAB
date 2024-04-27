import numpy as np
import matplotlib.pyplot as plt


def inspect_data(x, y, show_first_n=4, x_test=None, y_test=None):
    print(f'{x.shape=}, {y.shape=}')
    if x_test is not None and y_test is not None:
        print(f'{x_test.shape=}, {y_test.shape=}')
    print(f'First {show_first_n} (x, y) data pairs:')
    for xi, yi in zip(x[:show_first_n], y[:show_first_n]):
        print(f'{xi} => {yi}')
        

def visualize_activation_function(func):
    logits = np.linspace(-10, 10, 1001)
    
    plt.axhline(y=0, color="gray", linestyle="-")
    plt.axhline(y=1.0, color="gray", linestyle="--")
    plt.axhline(y=-1.0, color="gray", linestyle="--")
    plt.axvline(color="gray")
    plt.plot(logits, func(logits), 'r-', linewidth=3)
    plt.ylim(-1.2, 1.2)
    plt.title(f'Function: {func.__name__}'); plt.xlabel('logits'); plt.ylabel('y');
    plt.show()


def x_data_from_grid(min_xy, max_xy, grid_size, dtype=np.float32):
    values = np.linspace(min_xy, max_xy, grid_size, dtype=dtype)
    X1, X2 = np.meshgrid(values, values)
    return np.hstack((X1.reshape((X1.size, 1)), X2.reshape((X2.size, 1)))).astype(dtype)


def plot_data(x, y, ax=None, plot_xy_range=None, x_grid=None, y_grid=None,
              add_bar=True, axes_lines=True, x_test=None, y_test=None,
              x1_label='x1', x2_label='x2', title='Input space', do_show=True):

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    plt.cla()

    if plot_xy_range is not None:
        xlims = ylims = np.array(plot_xy_range)
    else:
        xlims = ylims = np.array([np.min(x[:, 0]), np.max(x[:, 1])])

    if x_grid is not None and y_grid is not None:
        grid_size = int(np.sqrt(x_grid.shape[0]))
        x1_grid = x_grid[:, 0].reshape((grid_size, grid_size))
        x2_grid = x_grid[:, 1].reshape((grid_size, grid_size))
        y_grid = y_grid.reshape((grid_size, grid_size))
        CS = ax.contourf(x1_grid, x2_grid, y_grid, np.linspace(0., 1., 51),
                         alpha=.8, cmap=plt.cm.get_cmap("bwr"))
        decision_line = plt.contour(CS, levels=[0.5], colors='k')
        if add_bar:
            cbar = plt.colorbar(CS)
            cbar.add_lines(decision_line)
            cbar.ax.set_ylabel('output value')

    if axes_lines is True:
        plt.axhline(y=0, color="gray", linestyle=":")
        plt.axvline(x=0, color="gray", linestyle=":")

    ax.scatter(x[:, 0], x[:, 1], c=y.flatten(), s=40, cmap=plt.cm.get_cmap("bwr"), edgecolors='k',
               label='Kółka: Train')

    if x_test is not None and y_test is not None:
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test.flatten(), marker='s',
                   s=40, cmap=plt.cm.get_cmap("bwr"), edgecolors='k', label='Kwadraty: Test')
        plt.legend()
    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.title(title)
    if do_show:
        plt.show()


def plot_two_layer_activations(model, x, y):
    x_grid = x_data_from_grid(min_xy=-1, max_xy=2, grid_size=1000)
    h_data = model.hidden_layer.forward(x)
    h_grid = model.hidden_layer.forward(x_grid)
    y_pred_grid = model.forward(x_grid)
    y_in_hidden_grid = model.output_layer.forward(x_grid)

    fig, axs = plt.subplots(2, 2)
    plot_data(x, y, ax=axs[0, 0], x_grid=x_grid, y_grid=h_grid[:, 0], plot_xy_range=[-1, 2], do_show=False,
              title='h1', add_bar=False)
    plot_data(x, y, ax=axs[0, 1], x_grid=x_grid, y_grid=h_grid[:, 1], plot_xy_range=[-1, 2], do_show=False,
              title='h2', add_bar=False)
    plot_data(h_data, y, ax=axs[1, 0], x_grid=x_grid, y_grid=y_in_hidden_grid, plot_xy_range=[-1, 2], do_show=False,
              x1_label='h1', x2_label='h2', title='y in hidden space', add_bar=False)
    plot_data(x, y, ax=axs[1, 1], x_grid=x_grid, y_grid=y_pred_grid, plot_xy_range=[-1, 2], do_show=False,
              title='y in input space', add_bar=False)
    for ax in axs.flatten():
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()
