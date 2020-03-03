import pyspark
import sklearn.cluster
import scipy
import numpy as np


def parseCSV(filename):
    with open(filename) as f:
        datafile = f.read()
        lines = datafile.split("\n")
        X = []
        for l in lines:
            fs = [float(x) for x in l.split(",")]
            X.append(fs)
        Xnp = np.array(X)
        return Xnp


def logsumexp(ns):
    max = np.max(ns)
    if np.isneginf(max):
        return float("-inf")
    ds = ns - max
    sumOfExp = np.exp(ds).sum()
    return max + np.log(sumOfExp)


def lognormalize_gamma(g):
    # TODO: scipy
    a = scipy.special.logsumexp(g, axis=1)
    # a = np.logaddexp.reduce(x)
    g_norm = g - a.reshape(-1, 1)
    return np.exp(g_norm)


def log_likelihood(X, k, means, cov):
    """
    compute log likelihood for every datapoint of every possible state
    TODO: vectorize ?
    """
    ll = np.zeros((len(X), k))
    for i in range(len(X)):
        for j in range(k):
            # TODO: scipy implement myself ?
            likel = scipy.stats.norm.pdf(X[i], means[j], np.sqrt(cov[j]))
            ll[i, j] = np.log(likel)

    return ll


def forward(loglikelihood, start, transition):
    """
    perform forward pass to compute alpha
    TODO: matrix faster ?
    """
    n, k = loglikelihood.shape
    with np.errstate(divide="ignore"):
        logstart = np.log(start)
        logtrans = np.log(transition)
    alpha = np.zeros((n, k))
    temp = np.zeros(k)

    for i in range(k):
        alpha[0, i] = logstart[i] + loglikelihood[0, i]

    for t in range(1, n):
        for j in range(k):
            for i in range(k):
                temp[i] = alpha[t-1, i] + logtrans[i, j]
            # pylint: disable=no-member
            # alpha[t, j] = np.logaddexp.reduce(temp) + loglikelihood[t, j]
            alpha[t, j] = logsumexp(temp) + loglikelihood[t, j]
    return alpha


def backward(loglikelihood, transition):
    """
    perform backward pass to compute beta
    """
    n, k = loglikelihood.shape
    with np.errstate(divide="ignore"):
        logtrans = np.log(transition)
    beta = np.zeros((n, k))
    temp = np.zeros(k)

    for i in range(k):
        beta[-1, i] = 0.0

    for t in range(n-2, -1, -1):
        for i in range(k):
            for j in range(k):
                temp[j] = logtrans[i, j] + loglikelihood[t+1, j] + beta[t+1, j]
            beta[t, i] = logsumexp(temp)
    return beta


def compute_trans(a, b, ll, transition):
    n, k = ll.shape
    with np.errstate(divide="ignore"):
        logtrans = np.log(transition)
    logxisum = np.full((k, k), float("-inf"))
    denom = logsumexp(a[-1])

    for t in range(n-1):
        for i in range(k):
            for j in range(k):
                logxi = a[t, i] + logtrans[i, j] + \
                    ll[t+1, j] + b[t+1, j] - denom
                logxisum[i, j] = logsumexp([logxisum[i, j], logxi])
    return logxisum


def normalize(a, axis=None):
    """
    normalize along axis
    making sure not to divide by zero
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    return a / a_sum


def update_params(sumgamma, xgammasum, xgammasumsquared, start, trans):
    norm_trans = normalize(trans, axis=1)
    norm_start = normalize(start)

    means = xgammasum / sumgamma
    # cov is based on newly computed means
    num = (means**2 * sumgamma - 2 * means *
           xgammasum + xgammasumsquared)

    # is prior necessary ?
    cov = (num + 0.01) / np.maximum(sumgamma, 1e-5)

    return norm_start, norm_trans, means, cov


def process(dataobj):
    X = dataobj["data"]
    means = dataobj["means"]
    cov = dataobj["cov"]
    startprop = dataobj["startprop"]
    transmat = dataobj["transmat"]
    k = dataobj["components"]

    n = len(X)

    loglikelihood = log_likelihood(X, k, means, cov)

    # forward pass
    logalpha = forward(loglikelihood, startprop, transmat)
    prop = logsumexp(logalpha[-1])

    # backward pass
    logbeta = backward(loglikelihood, transmat)

    loggamma = logalpha + logbeta
    gamma = lognormalize_gamma(loggamma)

    e_start = gamma[0]
    e_sumgamma = np.einsum("ij->j", gamma)
    e_xgammasum = np.dot(gamma.T, X)
    e_xgammasumsquared = np.dot(gamma.T, X**2)

    e_trans = np.zeros((k, k))
    if n > 1:
        logtrans = compute_trans(
            logalpha, logbeta, loglikelihood, transmat)
        e_trans = np.exp(logtrans)

    resultobj = {}
    resultobj["e_start"] = e_start
    resultobj["e_sumgamma"] = e_sumgamma
    resultobj["e_xgammasum"] = e_xgammasum
    resultobj["e_xgammasumsquared"] = e_xgammasumsquared
    resultobj["e_trans"] = e_trans
    resultobj["prop"] = prop

    return resultobj


def sum_intem_values(a, c):
    resultobj = {}
    resultobj["e_start"] = a["e_start"] + c["e_start"]
    resultobj["e_sumgamma"] = a["e_sumgamma"] + c["e_sumgamma"]
    resultobj["e_xgammasum"] = a["e_xgammasum"] + c["e_xgammasum"]
    resultobj["e_xgammasumsquared"] = a["e_xgammasumsquared"] + \
        c["e_xgammasumsquared"]
    resultobj["e_trans"] = a["e_trans"] + c["e_trans"]
    resultobj["prop"] = a["prop"] + c["prop"]
    return resultobj


def init(components, X):
    # join sequences if necessary
    if X.ndim == 2:
        X = np.concatenate(X)
    # initial means using kmeans
    kmeans = sklearn.cluster.KMeans(n_clusters=components)
    kmeans.fit(X.reshape(-1, 1))
    means = kmeans.cluster_centers_.reshape(-1)

    # initial covariance
    covar = np.cov(X)
    cov = np.tile(covar, components)
    # init start probablity
    startprop = np.tile(1/components, components)
    # init transition matrix
    transmat = np.tile(1/components, components **
                       2).reshape(components, components)
    return means, cov, startprop, transmat


def fit(X, components, iterations):
    with pyspark.SparkContext("local", "HMM") as sc:
        means, cov, startprop, transmat = init(components, X)

        for i in range(iterations):
            dataobjs = []
            for A in X:
                obj = {}
                obj["data"] = A
                obj["means"] = means
                obj["cov"] = cov
                obj["startprop"] = startprop
                obj["transmat"] = transmat
                obj["components"] = components
                dataobjs += [obj]

            # parallelize into 2 parts
            pardataobjs = sc.parallelize(dataobjs, 2)

            processed = pardataobjs.map(process)

            interm = processed.reduce(sum_intem_values)
            e_start = interm["e_start"]
            e_sumgamma = interm["e_sumgamma"]
            e_xgammasum = interm["e_xgammasum"]
            e_xgammasumsquared = interm["e_xgammasumsquared"]
            e_trans = interm["e_trans"]
            prop = interm["prop"]
            print(prop)

            startprop, transmat, means, cov = update_params(
                e_sumgamma, e_xgammasum, e_xgammasumsquared, e_start, e_trans)
        return startprop, transmat, means, cov


X = parseCSV("data.csv")
iterations = 10
components = 4
startprop, transmat, means, cov = fit(X, components, iterations)

print("startprop:")
print(startprop)
print("transmat:")
print(transmat)
print("means:")
print(means)
print("cov:")
print(cov)
