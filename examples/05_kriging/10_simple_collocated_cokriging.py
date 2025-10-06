r"""
Simple Collocated Cokriging
============================

Simple collocated cokriging is a variant of cokriging where only the
secondary variable collocated at the estimation location is considered.

This example uses the Markov Model I (MM1) approach where:

.. math:: C_{YZ}(h) = \rho_{YZ}(0) \cdot C_Z(h)

The MM1 cokriging estimator is:

.. math:: Z_{SCCK}^*(x_0) = Z_{SK}^*(x_0) \cdot (1 - k \cdot \lambda_{Y_0}) + \lambda_{Y_0} \cdot (Y(x_0) - m_Y) + m_Z

where :math:`k = C_{YZ}(0) / C_Z(0)`, :math:`\lambda_{Y_0}` is the collocated weight,
:math:`m_Y` is the secondary mean, and :math:`m_Z` is the primary mean.

Example
^^^^^^^

This example demonstrates SCCK with sparse primary data and dense secondary data
that shows clear spatial correlation, particularly useful in gap regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from gstools import Gaussian
from gstools.krige import Simple
from gstools.cokriging import SCCK

###############################################################################
# Generate data

np.random.seed(42)

# primary data - sparse sampling with gap around x=8-12
cond_pos = np.array([0.5, 2.1, 3.8, 6.2, 13.5])
cond_val = np.array([5.8, 6.2, 6.8, 6.1, 6.4])

# secondary data - dense sampling with strong spatial correlation
sec_pos = np.linspace(0, 15, 31)

# create secondary data correlated with primary pattern
primary_trend = np.interp(sec_pos, cond_pos, cond_val)

# add spatial feature in gap region (x=8-12) to demonstrate cokriging benefit
gap_feature = 0.4 * np.exp(-((sec_pos - 10.0) / 2.0)**2)
gap_feature2 = - 0.35 * np.exp(-((sec_pos - 4.0) / 2.0)**2)

# secondary = 0.85 * primary_pattern + gap_feature + small_noise
sec_val = 0.99 * primary_trend + gap_feature + gap_feature2 + \
    0.06 * np.random.randn(len(sec_pos))


# estimation grid
gridx = np.linspace(0.0, 15.0, 151)

###############################################################################
# Setup covariance model

model = Gaussian(dim=1, var=0.5, len_scale=2.0)

###############################################################################
# Simple Kriging

sk = Simple(
    model=model,
    cond_pos=cond_pos,
    cond_val=cond_val,
    mean=6.0
)
sk_field, sk_var = sk(pos=gridx, return_var=True)

###############################################################################
# Simple Collocated Cokriging

# calculate cross-correlation
sec_at_primary = np.interp(cond_pos, sec_pos, sec_val)
cross_corr = np.corrcoef(cond_val, sec_at_primary)[0, 1]

# calculate secondary mean (required for proper SCCK)
secondary_mean = np.mean(sec_val)

scck = SCCK(
    model=model,
    cond_pos=cond_pos,
    cond_val=cond_val,
    cross_corr=cross_corr,
    secondary_var=np.var(sec_val),
    mean=6.0,  # primary mean (mZ)
    secondary_mean=secondary_mean,  # secondary mean (mY)
)

# interpolate secondary data to grid
sec_grid = np.interp(gridx, sec_pos, sec_val)
scck_field, scck_var = scck(
    pos=gridx, secondary_data=sec_grid, return_var=True)

###############################################################################
# Results

print(f"Cross-correlation: {cross_corr:.3f}")
print(f"Primary mean: {6:.3f}")
print(f"Secondary mean: {secondary_mean:.3f}")
gap_mask = (gridx >= 8) & (gridx <= 12)
gap_improvement = np.mean(np.abs(scck_field[gap_mask] - sk_field[gap_mask]))
print(f"Mean difference in gap region: {gap_improvement:.3f}")

###############################################################################
# Plotting

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# plot data
ax1.scatter(cond_pos, cond_val, color="red",
            s=80, zorder=10, label="Primary data")
ax1.plot(sec_pos, sec_val, "b-", alpha=0.7, label="Secondary data")
ax1.axvspan(8, 12, alpha=0.2, color="orange", label="Gap region")
ax1.set_title("Data: Primary (sparse) vs Secondary (dense)")
ax1.set_ylabel("Value")
ax1.legend()
ax1.grid(True, alpha=0.3)

# plot kriging results
ax2.plot(gridx, sk_field, "r-", linewidth=2, label="Simple Kriging")
ax2.plot(gridx, scck_field, "b-", linewidth=2,
         label="Simple Collocated Cokriging")
ax2.scatter(cond_pos, cond_val, color="k", s=60, zorder=10, label="Conditions")
ax2.axvspan(8, 12, alpha=0.2, color="orange", label="Gap region")
ax2.set_title("Comparison: Simple Kriging vs Simple Collocated Cokriging")
ax2.set_xlabel("x")
ax2.set_ylabel("Value")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
