# Author: Elvis Dohmatob <gmdopp@gmail.com>

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.testing import assert_array_equal, assert_equal
import torch
from torch import nn

sys.path.append("/home/elvis/CODE/FORKED/elvis_dohmatob/rsfmri2tfmri")
from nn import _MLP


def _to_var(x, **kwargs):
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, **kwargs)


def _compute_ground_cost(x, y, p=2):
    if p == 2.:
        n = x.size(0)
        m = y.size(0)
        x2 = (x ** 2).sum(1)
        y2 = (y ** 2).sum(1).view(1, m)
        ground_cost = x2.repeat(1, m)
        ground_cost.add_(y2.repeat(n, 1))
        ground_cost.add_(-2 * x.mm(y.t()))
        return ground_cost
    else:
        raise NotImplementedError(
            "Ground cost not implemented for p=%s" % p)


def log_sum_exp(vec):
    """Compute log sum exp in a numerically stable way for the forward algorithm
    """
    max_score = vec[0, torch.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(
        torch.exp(vec - max_score_broadcast)))


def sinkhorn_stabilized(a, b, M, reg, numItermax=1000, tau=1e3,
                        stopThr=1e-9, warmstart=None, verbose=False,
                        print_period=20, log=False, **kwargs):
    # a = np.asarray(a, dtype=np.float64)
    # b = np.asarray(b, dtype=np.float64)
    # M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # test if multiple target
    if len(b.shape) > 1:
        nbb = b.shape[1]
        a = a[:, np.newaxis]
    else:
        nbb = 0

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = np.zeros(na), np.zeros(nb)
    else:
        alpha, beta = warmstart

    if nbb:
        u, v = np.ones((na, nbb)) / na, np.ones((nb, nbb)) / nb
    else:
        u, v = np.ones(na) / na, np.ones(nb) / nb

    def get_K(alpha, beta):
        """log space computation"""
        return np.exp(
            -(M - alpha.reshape((na, 1)) - beta.reshape((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return np.exp(-(M - alpha.reshape((na, 1)) - beta.reshape(
            (1, nb))) / reg + np.log(u.reshape((na, 1))) + np.log(
                v.reshape((1, nb))))

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (np.dot(K.T, u) + 1e-16)
        u = a / (np.dot(K, v) + 1e-16)

        # remove numerical problems and store them in K
        if np.abs(u).max() > tau or np.abs(v).max() > tau:
            if nbb:
                alpha, beta = alpha + reg * \
                    np.max(np.log(u), 1), beta + reg * np.max(np.log(v))
            else:
                alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
                if nbb:
                    u, v = np.ones((na, nbb)) / na, np.ones((nb, nbb)) / nb
                else:
                    u, v = np.ones(na) / na, np.ones(nb) / nb
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                    np.sum((v - vprev)**2) / np.sum((v)**2)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    # print('err=',err,' cpt=',cpt)
    if log:
        log['logu'] = alpha / reg + np.log(u)
        log['logv'] = beta / reg + np.log(v)
        log['alpha'] = alpha + reg * np.log(u)
        log['beta'] = beta + reg * np.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        if nbb:
            res = np.zeros((nbb))
            for i in range(nbb):
                res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if nbb:
            res = np.zeros((nbb))
            for i in range(nbb):
                res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
            return res
        else:
            return get_Gamma(alpha, beta, u, v)


def sinkhorn_epsilon_scaling(a, b, M, reg, n_iter=100, epsilon0=1e4,
                             n_inner_iter=100, tau=1e3, tol=1e-9,
                             warmstart=None, verbose=False, print_period=10,
                             log=False, **kwargs):
    """
    Solve the entropic regularization optimal transport problem with log
    stabilization and epsilon scaling.

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:
    `\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [2]_ but with the log stabilization
    proposed in [10]_ and the log scaling proposed in [9]_ algorithm 3.2


    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    tau : float
        thershold for max value in u or v for log scaling
    tau : float
        thershold for max value in u or v for log scaling
    warmstart : tible of vectors
        if given then sarting values for alpha an beta log scalings
    n_iter : int, optional
        Max number of iterations
    n_inner_iter : int, optional
        Max number of iterationsin the inner slog stabilized sinkhorn
    epsilon0 : int, optional
        first epsilon regularization value (then exponential decrease to reg)
    tol : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.bregman.sinkhorn_epsilon_scaling(a,b,M,1)
    array([[ 0.36552929,  0.13447071],
           [ 0.13447071,  0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation
    of Optimal Transport, Advances in Neural Information Processing
    Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms
    for Entropy Regularized Transport Problems. arXiv preprint
    arXiv:1610.06519.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    # a = np.asarray(a, dtype=np.float64)
    # b = np.asarray(b, dtype=np.float64)
    # M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    na = len(a)
    nb = len(b)

    # nrelative umerical precision with 64 bits
    numItermin = 35
    n_iter = max(numItermin, n_iter)  # ensure that last velue is exact

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = np.zeros(na), np.zeros(nb)
    else:
        alpha, beta = warmstart

    def get_K(alpha, beta):
        """log space computation"""
        return np.exp(
            -(M - alpha.reshape((na, 1)) - beta.reshape((1, nb))) / reg)

    # print(np.min(K))
    def get_reg(n):  # exponential decreasing
        return (epsilon0 - reg) * np.exp(-n) + reg

    loop = 1
    cpt = 0
    err = 1
    while loop:
        regi = get_reg(cpt)

        G, logi = sinkhorn_stabilized(
            a, b, M, regi, n_iter=n_inner_iter, tol=1e-9,
            warmstart=(alpha, beta), verbose=False, print_period=20, tau=tau,
            log=True)

        alpha = logi['alpha']
        beta = logi['beta']

        if cpt >= n_iter:
            loop = False

        if cpt % (print_period) == 0:  # spsion nearly converged
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = G
            err = np.linalg.norm(
                (np.sum(transp, axis=0) - b)) ** 2 + np.linalg.norm(
                    (np.sum(transp, axis=1) - a)) ** 2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 10) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= tol and cpt > numItermin:
            loop = False

        cpt = cpt + 1
    # print('err=',err,' cpt=',cpt)
    if log:
        log['alpha'] = alpha
        log['beta'] = beta
        log['warmstart'] = (log['alpha'], log['beta'])
        return G, log
    else:
        return G


def sinkhorn(ground_cost=None, x=None, y=None, epsilon=None, p=None,
             q=None, n_iter=10, tol=1e-4, check_tol_freq=10, return_errs=False,
             verbose=1):
    """Pytorch implementation of Cuturi's algorithm for entropic
    regularized Wasserstein distance.

    Parameters
    ==========
    ground_cost: 2d pytorch variable of numpy array, size / shape (n, m)
        Ground cost matrix; ground_cost[i, j] is the cost of moving a point
        x_i to a point y_j. If not provided, we'll compute an l2 loss matrix
        from the data x, y.

    epsilon : float, optional (default None)
        Inverse temperature. The high-temperature corresponds to MMD (matching
        all momments) while the low-temperature limit corresponds to the
        original Kantorovich formulation. For positive epsilon, the algorithm
        converges linearly to a unique solution (due to strict convexity,
        the rate of converging detoriating at low temperatures.

        If no value if provided, then we use a default value which is 10x
        the maxium ground cost between two points.

    n_iter : int, optional (default 10)
        The total number of iterations to run.

    """
    # compute cost matrix
    if ground_cost is None:
        if isinstance(x, np.ndarray):
            x = _to_var(torch.from_numpy(x)).float()
        if isinstance(y, np.ndarray):
            y = _to_var(torch.from_numpy(y)).float()
        ground_cost = _compute_ground_cost(x, y)
    elif isinstance(ground_cost, np.ndarray):
        ground_cost = _to_var(torch.from_numpy(ground_cost)).float()

    # if epsilon is None:
    #     epsilon = ground_cost.data.max() * epsilon_factor

    # # compute Gibbs kernel
    # kernel = torch.exp(-ground_cost / epsilon)

    # marginals
    n, m = ground_cost.size()
    if p is None:
        p = _to_var(torch.ones(n) / n)
    if q is None:
        q = _to_var(torch.ones(m) / m)

    ground_cost = ground_cost.data.numpy()
    # p = p.data.numpy()
    q = q.data.numpy()
    out = sinkhorn_epsilon_scaling(
        p, q, ground_cost, epsilon, n_iter=n_iter, tol=tol, log=return_errs)
    if return_errs:
        gamma = out[0]
        errs = out[1]["err"]
    else:
        gamma = out
    min_cost = sum(gamma * ground_cost)
    if return_errs:
        return gamma, min_cost, errs
    else:
        return gamma, _to_var(torch.from_numpy(min_cost))


def test_compute_ground_cost(random_state=42):
    rng = check_random_state(random_state)
    x = _to_var(torch.from_numpy(rng.randn(10, 3)))
    y = _to_var(torch.from_numpy(rng.randn(12, 3)))
    cost = _compute_ground_cost(x, y).data.numpy()
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            assert_equal(cost[i, j], sum((xi - yj) ** 2))


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
                            [8, 9, 98, 23.]])
    n_iter = 10
    for epsilon in np.logspace(-5, 0, num=6):
        assignment, _ = sinkhorn(ground_cost=ground_cost, n_iter=n_iter,
                                 epsilon=epsilon)
        assert_array_equal(assignment.argmax(axis=1), [2, 1, 0, 3])


class SinkhornGAN(nn.Module):
    def __init__(self, generator, L=10, epsilon=.1):
        super(SinkhornGAN, self).__init__()
        self.generator = generator
        self.L = L
        self.epsilon = epsilon

    def forward(self, z, y):
        x = self.generator(z)
        _, e = sinkhorn(x=x, y=y, epsilon=self.epsilon, n_iter=self.L,
                        return_errs=None, verbose=1)
        return e


def sinkhorn_experiments():
    rng = check_random_state(1)
    sizes = 300, 200

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

    epsilon = .01
    n_iter = 1
    pi, _, errs, _ = sinkhorn(x=x.T, y=y.T, epsilon=epsilon,
                              n_iter=n_iter, return_errs=True)
    pi = pi.data.numpy()

    if len(errs) > 1:
        plt.figure()
        plt.loglog(errs)
        plt.xlabel("iteration count")
        plt.ylabel("$||\\pi_x - p||_1$")
        plt.grid("on")

    plotp()
    A = pi * (pi > np.min(1. / np.asarray(sizes)) * .2)
    i, j = np.where(A != 0)
    plt.plot([x[0, i], y[0, j]], [x[1, i], y[1, j]], 'k', lw=2)

    A = pi * (pi > np.min(1. / np.asarray(sizes)) * .05)
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


def mnist_experiments():
    from sklearn.datasets import fetch_mldata

    mnist = fetch_mldata("MNIST original")

    # rescale the data, use the traditional train/test split
    images, labels = mnist.data, mnist.target.astype(np.int)
    images = images / 255.
    images = images[labels == 7]

    # N.B.: We need entropy in the real-world in order for a stochastic descent
    # algorithm to work
    images = images
    rng = check_random_state(42)
    perm = rng.permutation(len(images))
    images = images[perm]
    labels = labels[perm]

    image_shape = (28, 28)
    image_size = np.prod(image_shape)
    batch_size = 100
    n_epochs = 5
    images_train, images_test = images[:-batch_size], images[-batch_size:]

    low_dim = 2
    generator_hidden_dims = (500,)
    generator = _MLP((low_dim,) + generator_hidden_dims + (image_size,),
                     activation="relu", no_output_activation=True)
    print(generator)
    shgan = SinkhornGAN(generator, epsilon=100, L=100)

    def sample_noise(size):
        noise = _to_var(torch.rand(size, low_dim))
        return noise

    n_iter = len(images_train) // batch_size
    print_freq = 100
    print_freq = min(print_freq, n_iter)
    optimizer = torch.optim.Adam(shgan.generator.parameters(), lr=1e-5)
    images_test = _to_var(torch.from_numpy(images_test)).float()
    losses = []
    for epoch in range(n_epochs):
        print("\nEpoch %02i/%02i" % (epoch + 1, n_epochs))
        for it, batch in enumerate(gen_batches(len(images_train), batch_size)):
            # convert minibatch data to pytorch variables
            images_batch = images_train[batch]
            images_batch = _to_var(torch.from_numpy(images_batch)).float()
            z_batch = sample_noise(len(images_batch))
            loss = shgan.forward(z_batch, images_batch)
            losses.append(loss.data[0])

            shgan.generator.zero_grad()
            loss.backward()
            optimizer.step()

            # for p in shgan.parameters():
            #     p.data.clamp_(-.1, .1)

            if it % print_freq == 0:
                shgan.epsilon *= .9
                print("Batch [%03i -- %03i]/%03i loss=%g" % (
                    it, it + print_freq, n_iter, loss.data[0]))

                z_test = sample_noise(len(images_test))
                show_examples(images_test.data.numpy(),
                              image_shape=image_shape, n_cols=10)
                plt.title("true samples")
                plt.gray()

                # draw batch of hidden varibles from span of learned
                # representations
                shgan.generator.eval()
                images_test_fake = shgan.generator(z_test)
                show_examples(images_test_fake.data.numpy(),
                              image_shape=image_shape, n_cols=10)
                plt.title("fake samples")

                plt.show()
        plt.plot(losses)
        plt.grid("on")
        plt.show()

if __name__ == "__main__":
    if True:
        mnist_experiments()
    else:
        sinkhorn_experiments()
