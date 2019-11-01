import numpy as np
from numpy.linalg import solve, qr, cholesky
from typing import Tuple


from imbie2.const.lsq_methods import LSQMethod


def lscov(a, b, v: np.ndarray=None, dx: bool=False):
    """
    This is a python implementation of the matlab lscov function. This has been written based upon the matlab source
    code for lscov.m, which can be found here: http://opg1.ucsd.edu/~sio221/SIO_221A_2009/SIO_221_Data/Matlab5/Toolbox/matlab/matfun/lscov.m
    """

    m, n = a.shape
    if m < n:
        raise Exception("problem must be over-determined so that M > N.")
    if v is None:
        v = np.eye(m)

    if v.shape != (m, m):
        raise Exception("v must be a {0}-by-{0} matrix".format(m))

    qnull, r = qr(a, mode="complete")
    q = qnull[:, :n]
    r = r[:n, :n]

    qrem = qnull[:, n:]
    g = qrem.T.dot(v).dot(qrem)
    f = q.T.dot(v).dot(qrem)

    c = q.T.dot(b)
    d = qrem.T.dot(b)

    x = solve(r, (c - f.dot(solve(g, d))))

    # This was not required for merge_dM, and so has been removed as it has problems.
    if dx:
        u = cholesky(v).T
        z = solve(u, b)
        w = solve(u, a)
        mse = (z.T.dot(z) - x.T.dot(w.T.dot(z))) / (m - n)
        q, r = qr(w)
        ri = solve(r, np.eye(n)).T
        dx = np.sqrt(np.sum(ri * ri, axis=0) * mse).T

        return x, dx
    return x


def dm_to_dmdt(t: np.ndarray, dm: np.ndarray, sigma_dm: np.ndarray, wsize: float, truncate: bool=True,
               debug: bool=False, lsq_method: LSQMethod=LSQMethod.normal) -> Tuple[np.ndarray, np.ndarray]:
    """
    """

    dmdt = np.empty(t.shape, dtype=t.dtype)
    sigma_dmdt = np.empty(t.shape, dtype=t.dtype)
    model_fit_t = [None for _ in t]
    model_fit_dm = [None for _ in t]

    w_halfsize = wsize / 2.

    tmin = np.min(t)
    tmax = np.max(t)

    for i, it in enumerate(t):
        wmin = it - w_halfsize
        wmax = it + w_halfsize
        if wmin < tmin or wmax > tmax:

            if truncate:
                dmdt[i] = np.nan
                sigma_dmdt[i] = np.nan
                continue

            else:
                trunc_wmin = max(wmin, tmin)
                trunc_wmax = min(wmax, tmax)

                in_window = np.logical_and(t >= trunc_wmin, t < trunc_wmax).nonzero()
        else:
            in_window = np.logical_and(t >= wmin, t < wmax).nonzero()

        window_t = t[in_window]
        window_dm = dm[in_window]
        window_sigma_dm = sigma_dm[in_window]

        lsq_fit = np.vstack(
            [np.ones(window_t.shape, dtype=window_t.dtype), window_t]
        ).T

        if lsq_method == LSQMethod.regress:
            raise NotImplementedError()
        elif lsq_method == LSQMethod.normal:
            lsq_coef, lsq_se = lscov(lsq_fit, window_dm, dx=True)
        elif lsq_method == LSQMethod.weighted:
            w = np.diag(1. / np.square(window_sigma_dm))
            lsq_coef, lsq_se = lscov(lsq_fit, window_dm, w, dx=True)

        avg_window_sigma = np.sqrt(np.nanmean(window_sigma_dm ** 2.)) # / window_sigma_dm.size

        dmdt[i] = lsq_coef[1]
        # sigma_dmdt[i] = lsq_se[1]
        sigma_dmdt[i] = np.sqrt(
            lsq_se[1] ** 2 + avg_window_sigma ** 2
        )

        model_fit_t[i] = np.r_[wmin:wmax:.2]
        model_fit_dm[i] = lsq_coef[0] + lsq_coef[1] * model_fit_t[i]

    if debug:
        import matplotlib.pyplot as plt

        ax = plt.subplot(211)
        ax.set_xlim(np.min(t), np.max(t))

        ax.plot(t, dm, '.')
        for i in range(1, len(t), 5):
            if model_fit_t[i] is None or model_fit_dm[i] is None:
                continue
            ax.plot(model_fit_t[i], model_fit_dm[i], '--k')
        ax.plot(t, np.cumsum(dmdt / 12), 'r-')

        ax = plt.subplot(212)
        ax.set_xlim(np.min(t), np.max(t))
        ax.errorbar(t, dmdt, yerr=sigma_dmdt, color='b')

        plt.show()

    return dmdt, sigma_dmdt
