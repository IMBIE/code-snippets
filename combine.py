from .functions import *
from imbie2.const.error_methods import ErrorMethod


def weighted_combine(t, y, w=None, nsigma=None, average=False,
                     verbose=False, ret_data_out=False, error_method: ErrorMethod=ErrorMethod.average):
    """
    Combines a number of input sequences

    INPUTS:
        t: an iterable of the time-series arrays to be combined
        y: an iterable of the y-value arrays to be combined
        w (optional): an iterable of the weights to apply to each array
        nsigma: (optional) tolerance within which to consider a value to be valid
        average: (optional) if True, performs a moving average on the output
        verbose: (optional) if True, renders graphs for debugging purposes
        ret_data_out (optional): if True, returns the data_out array.
    OUTPUTS:
        t1: The abissca series of the combined data
        y1: The y-values of the combined data
        data_out (optional): returns the full data set
    """
    if not t or not y:
        return np.asarray([]), np.asarray([])
    if len(t) != len(y):
        return np.asarray([]), np.asarray([])
    if len(t) == 1:
        return np.asarray(t[0]), np.asarray(y[0])

    colors = ['r', 'g', 'b', 'c', 'y', 'm', 'o', 'k']
    if w is None:
        w = [1 for _ in t]
    for i, wi in enumerate(w):
        ti = t[i]
        if isinstance(wi, (float, int)):
            w[i] = wi * np.ones(ti.shape, dtype=np.float64)
        elif len(wi) != len(ti):
            raise ValueError("weights must be same length as time series")

    if verbose:
        for ti, yi, c in zip(t, y, colors[1:]):
            plt.plot(ti, yi, c+'-')
    # create _id array, which indicates which input array each element originated from
    _id = [np.ones(ti.shape, dtype=int)*(i+1) for i, ti in enumerate(t)]
    _id = np.concatenate(_id)
    # chain together input sequences
    t = np.concatenate(t)
    y = np.concatenate(y)
    w = np.concatenate(w)

    # sort the input time-values, and interpolate them to monthly values
    t1 = t2m(np.sort(t))
    # remove duplicates from where inputs have overlapped
    t1 = np.unique(t1)

    # create output array
    y1 = np.zeros(t1.shape, dtype=t1.dtype)
    # c1 is used to count the number of input data points that have been used for each output point
    c1 = np.zeros(t1.shape, dtype=np.float64)

    # create data_out array
    data_out = np.empty(
        (len(t1), np.max(_id) + 1),
        dtype=t1.dtype
    )
    # init. all values to NaN
    data_out.fill(np.NAN)

    data_out[:, 0] = t1
    for i in range(1, np.max(_id) + 1):
        # find valid data-points where the id matches the current input seq. being worked on
        ok = np.logical_and(
            _id == i, np.isfinite(y)
        )
        # ok = _id == i

        if nsigma is not None:
            # if nsigma has been specified, eliminate values far from the mean
            ok[ok] = np.abs(y[ok] - np.nanmean(y)) < max(nsigma, 1)*np.nanstd(y)
        # if we've eliminated all values in the current input, skip to the next one.
        if not ok.any():
            continue
        # get the valid items
        ti = t[ok]
        yi = y[ok]
        wi = w[ok]
        # sort by time
        o = np.argsort(ti)
        ti = ti[o]
        yi = yi[o]
        wi = wi[o]

        # match time to monthly values
        t2 = t2m(ti)
        # use interpolation to find y-values and weights at the new times
        y2 = interpol(ti, yi, t2)
        w2 = interpol(ti, wi, t2)
        # find locations where the times match other items in the input
        m1, m2 = match(np.floor(t1 * 12), np.floor(t2 * 12))
        # match,fix(t1*12),fix(t2*12),m1,m2
        # print repr(y1), repr(y2), repr(m1), repr(m2)

        # ok = np.isfinite(y2[m2])

        # m1 = m1[ok]
        # m2 = m2[ok]

        if verbose:
            plt.plot(t2, y2, colors[i]+'.')
        # add the values from the current input seq. to the output seq.
        try:
            if error_method == ErrorMethod.average or error_method == ErrorMethod.sum:
                y1[m1] += (y2[m2] * w2[m2])
            else:
                y1[m1] += (y2[m2] * w2[m2]) ** 2.
        except IndexError:
            print(m1, m2)
            raise
        data_out[m1, i] = y2[m2]
        # increment the values in c1 for each current point
        c1[m1] += w2[m2]
    # set any zeros in c1 to ones
    # c11 = np.maximum(c1, np.ones(c1.shape))
    c11 = c1.copy()
    c11[c11 == 0] = 1.
    # use c1 to calculate the element-wise average of the data
    if error_method == ErrorMethod.imbie1:
        # NB: this error method is included for comparability with IMBIE 1 IDL method,
        #     but is not recommended for normal use
        y1 = np.sqrt(y1 / c11) / np.sqrt(c11)
    elif error_method == ErrorMethod.rms:
        y1 = np.sqrt(y1 / c11)
    elif error_method == ErrorMethod.rss:
        y1 = np.sqrt(y1)
    elif error_method == ErrorMethod.average:
        y1 /= c11
    elif error_method != ErrorMethod.sum:
        raise ValueError("unrecognised error method: {}".format(error_method))

    # find any locations where no values were found
    ok = (c1 == 0)
    # set those locations to NaNs
    if ok.any():
        y1[ok] = np.NAN
    # optionally plot output
    if verbose:
        plt.plot(t1, y1, '--k')
    # optionally perform moving average
    if average:
        y1 = move_av(13./12, t1, y1)
        if verbose:
            plt.plot(t1, y1, '---k')
    data_out = data_out.T
    # render the plot
    if verbose:
        plt.show()
    # return the outputs
    if ret_data_out:
        return t1, y1, data_out
    return t1, y1
