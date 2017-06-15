"""
Copied from https://github.com/locuslab/icnn
"""
import numpy as np

def logistic(x):
    return 1. / (1. + np.exp(-x))

def logexp1p(x):
    """ Numerically stable log(1+exp(x))"""
    y = np.zeros_like(x)
    I = x>1
    y[I] = np.log1p(np.exp(-x[I]))+x[I]
    y[~I] = np.log1p(np.exp(x[~I]))
    return y

def proj_newton_logistic(A,b,lam0=None, line_search=True):
    """ minimize_{lam>=0, sum(lam)=1} -(A*1 + b)^T*lam + sum(log(1+exp(A^T*lam)))"""
    n = A.shape[0]
    c = np.sum(A,axis=1) + b
    e = np.ones(n)

    eps = 1e-12
    ALPHA = 1e-5
    BETA = 0.5

    if lam0 is None:
        lam = np.ones(n)/n
    else:
        lam = lam0.copy()

    for i in range(20):
        # compute gradient and Hessian of objective
        ATlam = A.T.dot(lam)
        z = 1/(1+np.exp(-ATlam))
        f = -c.dot(lam) + np.sum(logexp1p(ATlam))
        g = -c + A.dot(z)
        H = (A*(z*(1-z))).dot(A.T)

        # change of variables
        i = np.argmax(lam)
        y = lam.copy()
        y[i] = 1
        e[i] = 0

        g0 = g - e*g[i]
        H0 = H - np.outer(e,H[:,i]) - np.outer(H[:,i],e) + H[i,i]*np.outer(e,e)

        # compute bound set and Hessian of free set
        I = (y <= eps) & (g0 > 0)
        I[i] = True
        if np.linalg.norm(g0[~I]) < 1e-10:
            return lam
        d = np.zeros(n)
        H0_ = H0[~I,:][:,~I]
        try:
            d[~I] = np.linalg.solve(H0_, -g0[~I])
        except:
            # print('\n=== A\n\n', A)
            # print('\n=== H\n\n', H)
            # print('\n=== H0\n\n', H0)
            # print('\n=== H0_\n\n', H0_)
            # print('\n=== z\n\n', z)
            # print('\n=== iter: {}\n\n'.format(i))
            break
        # line search
        t = min(1. / np.max(abs(d)), 1.)
        for _ in range(10):
            y_n = np.maximum(y + t*d,0)
            y_n[i] = 1
            lam_n = y_n.copy()
            lam_n[i] = 1.-e.dot(y_n)
            if lam_n[i] >= 0:
                if line_search:
                    fn = -c.dot(lam_n) + np.sum(logexp1p(A.T.dot(lam_n)))
                    if fn < f + t*ALPHA*d.dot(g0):
                        break
                else:
                    break
            if max(t * abs(d)) < 1e-10:
                return lam_n
            t *= BETA

        e[i] = 1.
        lam = lam_n.copy()
    return lam

def solveBatch(fg, initXs, nIter=5, callback=None):
    """
    Warning: Comments are made by Vitchyr. I'm not sure if this is the
    correct interpretation.

    Note that fg should take a batch of x inputs. So initXs should be of
    shape (BATCH_SIZE, VAR_DIM).

    :param fg: fg(x) returns (value, gradient w.r.t. x)
        value shape: (BATCH_SIZE,)
        gradient shape: (BATCH_SIZE, VAR_DIM)
    :param initXs: initial x seed, shape (BATCH_SIZE, VAR_DIM)
    :param nIter: number of iterations
    :param callback: At each iteration, callback(t, fi) is called, where fi
    is the current value and t is the time
    :return: tuple (x, A, b, lam, xs, nIters)
        x = value
        A = gradients??
        b = ?
        lam = lambdas??
        xs = xs chosen during optimization
        nIters = the total number of iterations taken (one is taken per
        sample in the batch)
    """
    bsize = initXs.shape[0]
    A = [[] for i in range(bsize)]
    b = [[] for i in range(bsize)]
    xs = [[] for i in range(bsize)]
    lam = [None]*bsize

    x = initXs

    finished = []
    nIters = [nIter]*bsize

    finished = set()

    for t in range(nIter):
        fi, gi = fg(x)
        Ai = gi
        bi = fi - np.sum(gi * x, axis=1)
        if callback is not None:
            callback(t, fi)

        for u in range(bsize):
            if u in finished:
                continue

            A[u].append(Ai[u])
            b[u].append(bi[u])
            xs[u].append(np.copy(x[u]))

            prev_x = x[u].copy()
            if len(A[u]) > 1:
                lam[u] = proj_newton_logistic(np.array(A[u]), np.array(b[u]), None)
                x[u] = 1/(1+np.exp(np.array(A[u]).T.dot(lam[u])))
                x[u] = np.clip(x[u], 0.03, 0.97)

            else:
                lam[u] = np.array([1])
                x[u] = 1/(1+np.exp(A[u][0]))
                x[u] = np.clip(x[u], 0.03, 0.97)

            if max(abs((prev_x - x[u]))) < 1e-6:
                finished.add(u)

            A[u] = [y for i,y in enumerate(A[u]) if lam[u][i] > 0]
            b[u] = [y for i,y in enumerate(b[u]) if lam[u][i] > 0]
            xs[u] = [y for i,y in enumerate(xs[u]) if lam[u][i] > 0]
            lam[u] = lam[u][lam[u] > 0]

        if len(finished) == bsize:
            return x, A, b, lam, xs, nIters

    return x, A, b, lam, xs, nIters
