from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy.optimize import curve_fit

# Modified from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
def compute_scores(X, n_components):
    pca = PCA(whiten=True)

    pca_scores = []
    for n in range(1, n_components+1):
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))

    return pca_scores


# Obtained from the original UMAP implementation
# See https://github.com/lmcinnes/umap/blob/master/umap/umap_.py#L1386 
def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]