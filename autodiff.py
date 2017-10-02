# Author: Elvis Dohmatob <gmdopp@gmail.com>

from math import sqrt
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.testing import assert_array_equal, assert_almost_equal
import torch
from torch import nn
from bregman import sinkhorn_epsilon_scaling, _to_var
from mlp import MLP


def cdist(x, y, metric="sqeuclidean", p=None, backend="torch"):
    metric = metric.lower()
    backend = backend.lower()
    if backend == "torch":  # and torch.cuda.is_available():
        if metric == "sqeuclidean":
            if y is None:
                y = x
            x, y = map(_to_var, (x, y))
            n = x.size(0)
            m = y.size(0)
            x2 = (x ** 2).sum(1)
            y2 = (y ** 2).sum(1).view(1, m)
            ground_cost = x2.repeat(1, m)
            ground_cost.add_(y2.repeat(n, 1))
            ground_cost.add_(-2 * x.mm(y.t()))
        else:
            raise NotImplementedError(metric)
    elif backend == "custom":
        if metric == "sqeuclidean":
            if isinstance(x, torch.autograd.Variable):
                x = x.data.numpy()
            if isinstance(y, torch.autograd.Variable):
                y = y.data.numpy()
            n = len(x)
            m = len(y)
            ground_cost = np.tile((x ** 2).sum(1)[:, None], (1, m))
            ground_cost += np.tile((y ** 2).sum(1), (n, 1))
            ground_cost -= 2. * np.dot(x, y.T)
            ground_cost = _to_var(ground_cost, requires_grad=True)
    elif backend == "scipy":
        if isinstance(x, torch.autograd.Variable):
            x = x.data.numpy()
        if isinstance(y, torch.autograd.Variable):
            y = y.data.numpy()
        if y is None:
            ground_cost = spatial.distance.pdist(x, y, metric=metric, p=p)
        else:
            ground_cost = spatial.distance.cdist(x, y, metric=metric)
        ground_cost = _to_var(ground_cost, requires_grad=True)
    else:
        raise NotImplementedError(backend)
    return ground_cost


def test_cdist(random_state=42):
    rng = check_random_state(random_state)
    x = rng.randn(10, 3)
    y = rng.randn(12, 3)
    for metric, p in zip(["sqeuclidean", "cityblock"], [2, 1]):
        for backend in ["torch", "scipy", "custom"]:
            if backend != "scipy" and metric != "sqeuclidean":
                continue
            cost = cdist(x, y, backend=backend, metric=metric,
                         p=p).data.numpy()
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    assert_almost_equal(cost[i, j], sum(np.abs(xi - yj) ** p))


def sinkhorn(x=None, y=None, ground_cost=None, mu=None, nu=None,
             metric="sqeuclidean", epsilon=1., tol=1e-4, n_iter=100,
             n_inner_iter=10, log=False, **kwargs):
    if ground_cost is None:
        ground_cost = cdist(x, y, metric=metric)
    out = sinkhorn_epsilon_scaling(
        mu, nu, ground_cost, epsilon, tol=tol, log=log, n_iter=n_iter,
        n_inner_iter=n_inner_iter, **kwargs)
    if log:
        gamma, log = out
    else:
        gamma = out
    min_cost = torch.sum(ground_cost * gamma)
    if log:
        return min_cost, gamma, log
    else:
        return min_cost, gamma


def test_job_assignment():
    """
    We consider an example where four jobs (J1, J2, J3, and J4) need to be
    executed by four workers (W1, W2, W3, and W4), one job per worker.
    The matrix below shows the cost of assigning a certain worker to a
    certain job. The objective is to minimize the total cost of the
    assignment.

    Source: http://www.hungarianalgorithm.com/examplehungarianalgorithm.php
    """
    ground_cost = np.array([[82, 83, 69, 92],
                            [77, 37, 49, 92],
                            [11, 69, 5, 86],
                            [8, 9, 98, 23.]], dtype=np.double)
    ground_cost = _to_var(torch.from_numpy(ground_cost))
    for epsilon in np.logspace(-5, 0, num=6):
        print(epsilon)
        _, assignment = sinkhorn(ground_cost=ground_cost, epsilon=epsilon,
                                 n_iter=10)
        assignment = assignment.data.numpy()
        assert_array_equal(assignment.argmax(axis=1), [2, 1, 0, 3])


class SinkhornLoss(nn.Module):
    def __init__(self, L=10, epsilon=1., metric="sqeuclidean"):
        super(SinkhornLoss, self).__init__()
        self.L = L
        self.epsilon = epsilon
        self.metric = metric

    def forward(self, x, y):
        e, _ = sinkhorn(x=x, y=y, epsilon=self.epsilon, n_iter=self.L,
                        metric=self.metric, verbose=1, n_inner_iter=1)
        return e


def sinkhorn_experiments():
    rng = check_random_state(1)
    sizes = 30, 20

    x = rng.rand(2, sizes[0]) - .5
    theta = 2 * np.pi * rng.rand(1, sizes[1])
    r = .8 + .2 * rng.rand(1, sizes[1])
    y = np.vstack((np.cos(theta) * r, np.sin(theta) * r))

    def plotp():
        plt.figure(figsize=(10, 10))
        for z, col in zip((x, y), "br"):
            plt.scatter(z[0, :], z[1, :], s=200, edgecolors="k",
                        c=col, linewidths=1)

        plt.axis("off")
        plt.xlim(np.min(y[0]) - .1, np.max(y[0]) + .1)
        plt.ylim(np.min(y[1]) - .1, np.max(y[1]) + .1)

    plotp()

    epsilon = .1
    n_iter = 1
    _, gamma, errs = sinkhorn(x=x.T, y=y.T, epsilon=epsilon,
                              n_iter=n_iter, log=True)
    gamma = gamma.data.numpy()
    if len(errs) > 1 and 0:
        plt.figure()
        plt.loglog(errs["err"])
        plt.xlabel("iteration count")
        plt.ylabel("err")
        plt.grid("on")

    plotp()
    A = gamma * (gamma > np.min(1. / np.asarray(sizes)) * 1e-4)
    i, j = np.where(A != 0)
    plt.plot([x[0, i], y[0, j]], [x[1, i], y[1, j]], 'k', lw=2)

    A = gamma * (gamma > np.min(1. / np.asarray(sizes)) * 7e-5)
    i, j = np.where(A != 0)
    plt.plot([x[0, i], y[0, j]], [x[1, i], y[1, j]], 'k:', lw=1)

    plt.show()


def show_examples(data, image_shape=None, n=None, n_cols=20):
    if n is None:
        n = len(data)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / float(n_cols)))
    if image_shape is None:
        assert data[0].ndim == 2
        image_shape = data[0].shape
    img_rows, img_cols = image_shape
    figure = np.zeros((img_rows * n_rows, img_cols * n_cols))
    for k, x in enumerate(data[:n]):
        r = k // n_cols
        c = k % n_cols
        figure[r * img_rows: (r + 1) * img_rows,
               c * img_cols: (c + 1) * img_cols] = x.reshape(image_shape)
    plt.figure(figsize=(12, 10))
    plt.imshow(figure)
    plt.axis("off")
    plt.tight_layout()


def mnist_experiments(semi_sup=False):
    from sklearn.datasets import fetch_mldata
    from sklearn.preprocessing import OneHotEncoder

    # load data
    mnist = fetch_mldata("MNIST original")

    # rescale the data, use the traditional train/test split
    images, labels = mnist.data, mnist.target.astype(np.int)
    images = images / 255.
    if True:
        msk = np.isin(labels, [0, 3, 6, 8])
        images = images[msk]
        labels = labels[msk]
    one_hot = OneHotEncoder().fit(labels[:, None])

    # N.B.: We need entropy in the real-world in order for a stochastic descent
    # algorithm to work
    rng = check_random_state(0)

    image_shape = (28, 28)
    image_size = np.prod(image_shape)
    batch_size = 200
    n_epochs = 20

    # MLP generator network
    low_dim = 2
    generator_hidden_dims = (256, 512)
    w_dim = low_dim
    if semi_sup:
        y_dim = len(np.unique(labels))
        w_dim += y_dim
    generator = MLP((w_dim,) + generator_hidden_dims + (image_size,),
                    activation="relu")
    generator.double()
    print(generator)

    # Sinkhorn loss network
    sinkhorn_loss = SinkhornLoss(epsilon=1000., L=10, metric="sqeuclidean")

    # Laten space sampler
    def sample_noise(size):
        return _to_var(torch.rand(size, low_dim)).double()

    # misc
    n_iter = len(images) // batch_size
    print_freq = 10
    print_freq = min(print_freq, n_iter)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)
    losses = []
    batches = list(gen_batches(len(images), batch_size))

    def set_regu(t):
        sinkhorn_loss.epsilon = 10. / sqrt(t)

    # training loop proper
    for epoch in range(n_epochs):
        print("\nEpoch %02i/%02i" % (epoch + 1, n_epochs))
        perm = rng.permutation(len(images))
        images = images[perm]
        labels = labels[perm]
        for it, batch in enumerate(batches):
            # sample
            images_batch = images[batch]
            images_batch = _to_var(torch.from_numpy(images_batch))
            if semi_sup:
                labels_batch = one_hot.transform(labels[batch][:, None])
                labels_batch = _to_var(labels_batch.toarray())
            else:
                labels_batch = None

            # sample back of noise vectors from latent space
            z_batch = sample_noise(len(images_batch))

            # try to fake
            if semi_sup:
                w_batch = torch.cat((z_batch, labels_batch), dim=1)
            else:
                w_batch = z_batch
            fake = generator(w_batch)

            # compute loss
            set_regu(len(losses) + 1)
            loss = sinkhorn_loss(fake, images_batch)
            losses.append(loss.data[0])

            # backprop the error and then step
            generator.zero_grad()
            loss.backward()
            optimizer.step()

            # report progress
            if it % print_freq == 0:
                print(
                    "Batch [%05i -- %05i]/%05i reg=%g, Sinkhorn loss=%g" % (
                        it * batch_size, (it + print_freq) * batch_size,
                        len(images), sinkhorn_loss.epsilon, loss.data[0]))

        # plot real images
        show_examples(images[:100], image_shape=image_shape, n_cols=10)
        plt.title("Epoch %i/%i: true samples" % (epoch, n_epochs))
        plt.gray()
        plt.tight_layout()

        # draw batch of noise vectors from latent space
        generator.eval()
        noise = sample_noise(100)
        if semi_sup:
            labels_fake = one_hot.transform(labels[:100][:, None])
            labels_fake = _to_var(labels_fake.toarray())
            w = torch.cat((noise, labels_fake), dim=1)
        else:
            w = noise
        images_fake = generator(w)
        show_examples(images_fake.data.numpy(), n_cols=10,
                      image_shape=image_shape)
        plt.title("Epoch %i/%i: fake samples" % (epoch + 1, n_epochs))
        plt.tight_layout()

        # plot convergence curve
        plt.figure()
        plt.plot(losses, linewidth=2)
        plt.xlabel("mini-batches")
        plt.ylabel("training loss")
        plt.grid("on")
        plt.show()

if __name__ == "__main__":
    if True:
        mnist_experiments(semi_sup=True)
    else:
        sinkhorn_experiments()
