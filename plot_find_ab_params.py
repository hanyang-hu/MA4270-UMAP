import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

# Test the function
spread, min_dist = 1.0, 1.0
a, b = find_ab_params(spread, min_dist)

# Plot the curve
xv = np.linspace(0, spread * 3, 300)
yv_ab = 1.0 / (1.0 + a * xv ** (2 * b)) # approximated curve
yv_sm = np.zeros(xv.shape)
yv_sm[xv < min_dist] = 1.0
yv_sm[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread) # original curve
plt.plot(xv, yv_ab, label='Approximated Curve')
plt.plot(xv, yv_sm, label='Offset Exponential Decay')
plt.legend()
plt.title('Approximated Curve vs. Offset Exponential Decay')
plt.xlabel('Distance')
plt.ylabel('Grade of Membership')
plt.show()

