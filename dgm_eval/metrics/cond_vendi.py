from sklearn import preprocessing
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import scipy
import scipy.linalg
import numpy as np
from tqdm import tqdm
from .conditional_vendi_utils import ConditionalEvaluation


def compute_vendi_alpha_score(X, q=1, normalize=True, kernel='linear', kernel_bandwidth=None):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    if kernel == 'linear':
        S = X @ X.T
    elif kernel == 'polynomial':
        S = polynomial_kernel(X, degree=3, gamma=None, coef0=1)  # currently hardcoding kernel params to match KID
    elif kernel == 'gaussian':
        if kernel_bandwidth is None:
            raise ValueError('Gaussian kernel used for Vendi score, but kernel bandwidth is None.')
        S = rbf_kernel(X, gamma=1 / (kernel_bandwidth ** 2))
        w = scipy.linalg.eigvalsh(S / n)
        output = np.exp(entropy_q(w, q=q))

        eval_model = ConditionalEvaluation(sigma=(kernel_bandwidth, kernel_bandwidth))
        ent = eval_model.compute_entropy(S, order=q).detach().cpu()
        return float(np.exp(ent).item())

    else:
        raise NotImplementedError("kernel not implemented")
    # print('similarity matrix of shape {}'.format(S.shape))
    w = scipy.linalg.eigvalsh(S / n)
    return np.exp(entropy_q(w, q=q))


def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_ ** q).sum()) / (1 - q)


