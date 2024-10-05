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

# Plot attractive and repulsive forces for a set of p_ij
def attractive_force(x, a, b, p):
    return 2 * a * b * p / (x * (a + x ** (-2 * b)))

def repulsive_force(x, a, b, p):
    return 2 * b * (1-p) / (x * (1 + a * x ** (2 * b)))

p_ij = [0.2, 0.5, 0.8]
color = ['r', 'g', 'b']
xv = np.linspace(0.00001, spread * 3, 300)
plt.ylim(0, 5)
for p in p_ij:
    yv_at = attractive_force(xv, a, b, p)
    yv_rp = repulsive_force(xv, a, b, p)
    plt.plot(xv, yv_at, label=f'Attractive Force p_ij={p}', color=color[p_ij.index(p)], linestyle='-')
    plt.plot(xv, yv_rp, label=f'Repulsive Force p_ij={p}', color=color[p_ij.index(p)], linestyle='--')
    x_intersect = xv[np.argmin(np.abs(yv_at-yv_rp))]
    y_intersect = yv_at[np.argmin(np.abs(yv_at-yv_rp))]
    plt.scatter(x_intersect, y_intersect, c=color[p_ij.index(p)])
plt.legend()
plt.title('Attractive and Repulsive Forces')
plt.xlabel('Distance')
plt.ylabel('Absolute Value of Force')
plt.show()


