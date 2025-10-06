r"""
Intrinsic Collocated Cokriging
===============================

Intrinsic Collocated Cokriging (ICCK) is an advanced cokriging variant that
improves upon Simple Collocated Cokriging (SCCK) by providing better variance
estimation and using secondary data at all primary conditioning locations.

Unlike SCCK's MM1 approach, ICCK uses the more accurate variance formula:

.. math:: \sigma^2_{ICCK} = (1 - \rho_0^2) \cdot \sigma^2_{SK}

where :math:`\rho_0^2 = C_{YZ}^2(0) / (C_Y(0) \cdot C_Z(0))` is the squared
correlation coefficient at zero lag.

The ICCK weights are:

.. math:: \lambda = \lambda_{SK}, \quad \mu = -\frac{C_{YZ}(0)}{C_Y(0)} \lambda_{SK}, \quad \lambda_{Y_0} = \frac{C_{YZ}(0)}{C_Y(0)}

Example
^^^^^^^

This example demonstrates ICCK vs SCCK, showing the improved variance behavior
and better handling of cross-correlated secondary information.
"""

import numpy as np
import matplotlib.pyplot as plt
from gstools import Gaussian
from gstools.krige import Simple
from gstools.cokriging import SCCK, ICCK

###############################################################################
# Generate data

np.random.seed(4)

# primary data - sparse sampling with gap around x=8-12
cond_pos = np.array([0.5, 2.1, 3.8, 6.2, 13.5])
cond_val = np.array([0.8, 1.2, 1.8, 2.1, 1.4])

# secondary data - dense sampling with strong spatial correlation
sec_pos = np.linspace(0, 15, 31)

# create secondary data correlated with primary pattern
primary_trend = np.interp(sec_pos, cond_pos, cond_val)

# add spatial feature in gap region (x=8-12) to demonstrate cokriging benefit
gap_feature = - 1.6 * np.exp(-((sec_pos - 10.0) / 2.0)**2)
gap_feature2 = - 0.95 * np.exp(-((sec_pos - 4.0) / 2.0)**2)

# secondary = 0.85 * primary_pattern + gap_feature + small_noise
sec_val = 0.99 * primary_trend + gap_feature + gap_feature2
# Secondary data at primary conditioning locations (required for ICCK)
sec_at_primary = np.interp(cond_pos, sec_pos, sec_val)

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
    mean=1.0
)
sk_field, sk_var = sk(pos=gridx, return_var=True)

###############################################################################
# Simple Collocated Cokriging (SCCK)

# calculate cross-correlation
cross_corr = np.corrcoef(cond_val, sec_at_primary)[0, 1]

# calculate secondary mean (required for proper cokriging)
secondary_mean = np.mean(sec_val)
print(secondary_mean)

scck = SCCK(
    model=model,
    cond_pos=cond_pos,
    cond_val=cond_val,
    cross_corr=cross_corr,
    secondary_var=np.var(sec_val),
    mean=1.0,  # primary mean
    secondary_mean=secondary_mean  # secondary mean for proper cokriging
)

# interpolate secondary data to grid
sec_grid = np.interp(gridx, sec_pos, sec_val)
scck_field, scck_var = scck(
    pos=gridx, secondary_data=sec_grid, return_var=True)

###############################################################################
# Intrinsic Collocated Cokriging (ICCK)

icck = ICCK(
    model=model,
    cond_pos=cond_pos,
    cond_val=cond_val,
    secondary_cond_pos=cond_pos,  # Secondary positions (same as primary)
    secondary_cond_val=sec_at_primary,  # Secondary values at primary locations
    cross_corr=cross_corr,
    secondary_var=np.var(sec_val),
    mean=1.0,  # primary mean
    secondary_mean=secondary_mean  # secondary mean for proper cokriging
)

icck_field, icck_var = icck(
    pos=gridx, secondary_data=sec_grid, return_var=True)

###############################################################################
# Results and Analysis

print(f"Cross-correlation: {cross_corr:.3f}")
gap_mask = (gridx >= 8) & (gridx <= 12)

# Compare field estimates in gap region
scck_gap_improvement = np.mean(
    np.abs(scck_field[gap_mask] - sk_field[gap_mask]))
icck_gap_improvement = np.mean(
    np.abs(icck_field[gap_mask] - sk_field[gap_mask]))

print(f"SCCK mean difference in gap region: {scck_gap_improvement:.3f}")
print(f"ICCK mean difference in gap region: {icck_gap_improvement:.3f}")

# Compare variance behavior
print(f"SK variance range: [{np.min(sk_var):.3f}, {np.max(sk_var):.3f}]")
print(f"SCCK variance range: [{np.min(scck_var):.3f}, {np.max(scck_var):.3f}]")
print(f"ICCK variance range: [{np.min(icck_var):.3f}, {np.max(icck_var):.3f}]")

# Theoretical correlation coefficient
C_Z0, C_Y0, C_YZ0 = icck._compute_covariances()
rho_squared = icck._compute_correlation_coeff_squared(C_Z0, C_Y0, C_YZ0)
print(f"Theoretical ρ₀²: {rho_squared:.3f}")
print(f"ICCK variance reduction factor: {1 - rho_squared:.3f}")

###############################################################################
# Plotting

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Data
ax1.scatter(cond_pos, cond_val, color="red",
            s=80, zorder=10, label="Primary data")
ax1.scatter(cond_pos, sec_at_primary, color="blue", s=60, zorder=9,
            marker="s", label="Secondary at primary")
ax1.plot(sec_pos, sec_val, "b-", alpha=0.7, label="Secondary data")
ax1.axvspan(8, 12, alpha=0.2, color="orange", label="Gap region")
ax1.set_title("Data: Primary and Secondary Variables")
ax1.set_ylabel("Value")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Field estimates comparison
ax2.plot(gridx, sk_field, "r-", linewidth=2, label="Simple Kriging")
ax2.plot(gridx, scck_field, "b-", linewidth=2, label="SCCK")
ax2.plot(gridx, icck_field, "g-", linewidth=2, label="ICCK")
ax2.scatter(cond_pos, cond_val, color="k", s=60, zorder=10, label="Conditions")
ax2.axvspan(8, 12, alpha=0.2, color="orange", label="Gap region")
ax2.set_title("Field Estimates: SK vs SCCK vs ICCK")
ax2.set_ylabel("Value")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Variance comparison
ax3.plot(gridx, sk_var, "r-", linewidth=2, label="SK variance")
ax3.plot(gridx, scck_var, "b-", linewidth=2, label="SCCK variance")
ax3.plot(gridx, icck_var, "g-", linewidth=2, label="ICCK variance")
ax3.axvspan(8, 12, alpha=0.2, color="orange", label="Gap region")
ax3.set_title("Variance Comparison")
ax3.set_ylabel("Variance")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Variance reduction in gap region
gap_sk_var = sk_var[gap_mask]
gap_scck_var = scck_var[gap_mask]
gap_icck_var = icck_var[gap_mask]
gap_x = gridx[gap_mask]

ax4.plot(gap_x, gap_sk_var, "r-", linewidth=3, label="SK variance")
ax4.plot(gap_x, gap_scck_var, "b-", linewidth=3, label="SCCK variance")
ax4.plot(gap_x, gap_icck_var, "g-", linewidth=3, label="ICCK variance")
ax4.fill_between(gap_x, gap_sk_var, alpha=0.3, color="red")
ax4.fill_between(gap_x, gap_icck_var, alpha=0.3, color="green")
ax4.set_title("Variance Reduction in Gap Region")
ax4.set_xlabel("x")
ax4.set_ylabel("Variance")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Summary

print("\n" + "="*60)
print("SUMMARY: ICCK vs SCCK Performance")
print("="*60)
print(f"Cross-correlation coefficient: {cross_corr:.3f}")
print(f"Theoretical variance reduction (1-ρ₀²): {1-rho_squared:.3f}")
print(f"")
print(f"Mean variance in gap region:")
print(f"  SK:   {np.mean(gap_sk_var):.4f}")
print(f"  SCCK: {np.mean(gap_scck_var):.4f}")
print(f"  ICCK: {np.mean(gap_icck_var):.4f}")
print(f"")
print(f"ICCK advantages:")
print(f"  - Improved variance estimation (no MM1 inflation)")
print(f"  - Mathematical consistency with correlation theory")
print(f"  - Better uncertainty quantification")
print(f"  - Uses all available secondary information")
