# -*- coding: utf-8 -*-
"""
Bregman projections for regularized OT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#         Elvis Dohmatob <gmdopp@gmail.com>
#
# License: MIT License

from math import exp
import numpy as np
import torch


def _to_var(x, **kwargs):
    if isinstance(x, torch.autograd.Variable):
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, **kwargs)


def _outer_sum(a, b):
    n, m = len(a), len(b)
    a = a.view(n, 1).repeat(1, m)
    b = b.view(1, m).repeat(n, 1)
    return a + b


def sinkhorn_stabilized(a, b, M, reg, n_iter=1000, tau=1e3, tol=1e-9,
                        warmstart=None, verbose=False, print_period=20,
                        log=False, **kwargs):
    """
    Solve the entropic regularization OT problem with log stabilization

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
    proposed in [10]_ an defined in [9]_ (Algo 3.1) .


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
    warmstart : tible of vectors
        if given then sarting values for alpha an beta log scalings
    n_iter : int, optional
        Max number of iterations
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
    >>> ot.bregman.sinkhorn_stabilized(a,b,M,1)
    array([[ 0.36552929,  0.13447071],
           [ 0.13447071,  0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
    Transport, Advances in Neural Information Processing Systems (NIPS) 26,
    2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for
    Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
    Scaling algorithms for unbalanced transport problems. arXiv preprint
    arXiv:1607.05816.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    if a is None:
        a = _to_var(torch.ones(M.size(0)) / M.size(0))
    if b is None:
        b = _to_var(torch.ones(M.size(1)) / M.size(1))
    a, b = a.double(), b.double()

    # test if multiple target
    if len(b.size()) > 1:
        nbb = b.size(1)
        a = a.view(len(a), 1)
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
        alpha, beta = torch.zeros(na), torch.zeros(nb)
    else:
        alpha, beta = warmstart
    alpha, beta = map(_to_var, (alpha, beta))
    alpha, beta = alpha.double(), beta.double()

    if nbb:
        u, v = torch.ones(na, nbb) / na, torch.ones(nb, nbb) / nb
    else:
        u, v = torch.ones(na) / na, torch.ones(nb) / nb
    u, v = map(_to_var, (u, v))
    u, v = u.double(), v.double()

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M.sub(_outer_sum(alpha, beta))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return torch.exp(-(M.sub(_outer_sum(alpha, beta))) / reg + _outer_sum(
            torch.log(u), torch.log(v)))

    # print(torch.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        # v = b / (np.dot(K.T, u) + 1e-16)
        v = b.div(K.t().mv(u).add(1e-16))
        # u = a / (np.dot(K, v) + 1e-16)
        u = a.div(K.mv(v).add(1e-16))

        # remove numerical problems and store them in K
        if torch.abs(u).max() > tau or torch.abs(v).max() > tau:
            if nbb:
                # alpha.add_(reg * torch.max(torch.log(u), 1))
                alpha = alpha + reg * torch.max(torch.log(u), 1)
                # beta.add_(reg * torch.max(torch.log(v)))
                beta = beta + reg * torch.max(torch.log(v))
            else:
                # alpha.add_(reg * torch.log(u))
                alpha = alpha + reg * torch.log(u)
                # beta.add_(reg * torch.log(v))
                beta = beta + reg * torch.log(v)
                if nbb:
                    u = torch.ones((na, nbb)) / na
                    v = torch.ones((nb, nbb)) / nb
                else:
                    u, v = torch.ones(na) / na, torch.ones(nb) / nb
            u, v = map(_to_var, (u, v))
            u, v = u.double(), v.double()
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = torch.sum((u - uprev) ** 2) / torch.sum((u) ** 2) + \
                      torch.sum((v - vprev) ** 2) / torch.sum((v) ** 2)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = torch.norm((torch.sum(transp, 0) - b)) ** 2
            err = err.data.numpy()[0]
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 20) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= tol:
            loop = False

        if cpt >= n_iter:
            loop = False

        if np.any(np.isnan(u.data.numpy())) or np.any(
                np.isnan(v.data.numpy())):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    # print('err=',err,' cpt=',cpt)
    if log:
        log['logu'] = alpha / reg + torch.log(u)
        log['logv'] = beta / reg + torch.log(v)
        log['alpha'] = alpha + reg * torch.log(u)
        log['beta'] = beta + reg * torch.log(v)
        log['warmstart'] = (log['alpha'], log['beta'])
        if nbb:
            res = torch.zeros(nbb)
            for i in range(nbb):
                res[i] = torch.sum(get_Gamma(alpha, beta, u[:, i],
                                             v[:, i]) * M)
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if nbb:
            res = torch.zeros(nbb)
            for i in range(nbb):
                res[i] = torch.sum(get_Gamma(alpha, beta, u[:, i],
                                            v[:, i]) * M)
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

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
    Transport, Advances in Neural Information Processing Systems (NIPS) 26,
    2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for
    Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    if a is None:
        a = _to_var(torch.ones(M.size(0)) / M.size(0))
    if b is None:
        b = _to_var(torch.ones(M.size(1)) / M.size(1))
    a, b = a.double(), b.double()

    # init data
    na = len(a)
    nb = len(b)

    # nrelative umerical precision with 64 bits
    # numItermin = 35
    # n_iter = max(numItermin, n_iter)  # ensure that last value is exact

    cpt = 0
    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = torch.zeros(na), torch.zeros(nb)
    else:
        alpha, beta = warmstart
    alpha, beta = map(_to_var, (alpha, beta))
    alpha, beta = alpha.double(), beta.double()

    def get_K(alpha, beta):
        """log space computation"""
        return torch.exp(-(M.sub(_outer_sum(alpha, beta))) / reg)

    # print(torch.min(K))
    def get_reg(n):  # exponential decreasing
        return float((epsilon0 - reg) * exp(-n) + reg)

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
            err = torch.norm(
                (torch.sum(transp, 0) - b)) ** 2 + torch.norm(
                    (torch.sum(transp, 1) - a)) ** 2
            err = err.data.numpy()[0]
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % (print_period * 10) == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        if err <= tol:  # and cpt > numItermin:
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


def geometric_bar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    assert(len(weights) == alldistribT.sisze(1))
    return torch.exp(torch.log(alldistribT).mv(weights))


def geometric_mean(alldistribT):
    """return the  geometric mean of distributions"""
    return torch.exp(torch.mean(torch.log(alldistribT), 1))


def barycenter(A, M, reg, weights=None, n_iter=1000, tol=1e-4, verbose=False,
               log=False):
    """Compute the entropic regularized wasserstein barycenter of distributions A

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = arg\min_\mathbf{a} \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
    distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{a}_i` are training distributions in the columns of matrix
    :math:`\mathbf{A}`
    - reg and :math:`\mathbf{M}` are respectively the regularization term and
    the cost matrix for OT

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [3]_

    Parameters
    ----------
    A : np.ndarray (d,n)
        n training distributions of size d
    M : np.ndarray (d,d)
        loss matrix   for OT
    reg : float
        Regularization term >0
    n_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    a : (d,) ndarray
        Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [3] Benamou, J. D., Carlier, G., Cuturi, M., Nenna, L., & Peyré, G.
    (2015). Iterative Bregman projections for regularized transportation
    problems. SIAM Journal on Scientific Computing, 37(2), A1111-A1138.

    """

    if weights is None:
        weights = torch.ones(A.size(1)) / A.size(1)
    else:
        assert(len(weights) == A.size(1))

    if log:
        log = {'err': []}

    K = torch.exp(-M / reg)

    cpt = 0
    err = 1

    UKv = torch.mv(K, torch.div(A.t(), torch.sum(K, 0)).t())
    u = (geometric_mean(UKv).div(UKv.t())).t()

    while (err > tol and cpt < n_iter):
        cpt = cpt + 1
        UKv = u * torch.mv(K, torch.div(A, K.mv(u)))
        u = (u.t() * geometric_bar(weights, UKv)).t().div(UKv)

        if cpt % 10 == 1:
            err = torch.sum(torch.std(UKv, 1))

            # log and verbose print
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    if log:
        log['niter'] = cpt
        return geometric_bar(weights, UKv), log
    else:
        return geometric_bar(weights, UKv)
