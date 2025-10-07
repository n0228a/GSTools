r"""
Intrinsic Collocated Cokriging
-------------------------------

Intrinsic Collocated Cokriging (ICCK) improves variance estimation
compared to Simple Collocated Cokriging.

The variance formula is:

.. math:: \sigma^2_{ICCK} = (1 - \rho_0^2) \cdot \sigma^2_{SK}

Example
^^^^^^^

Here we compare Simple Kriging with Intrinsic Collocated Cokriging.
"""

import matplotlib.pyplot as plt
import numpy as np

from gstools import Gaussian, krige
from gstools.cokriging import IntrinsicCollocated

# condtions
cond_pos = [0.3, 1.9, 1.1, 3.3, 4.7]
cond_val = [0.47, 0.56, 0.74, 1.47, 1.74]
# resulting grid
gridx = np.linspace(0.0, 15.0, 151)
# spatial random field class
model = Gaussian(dim=1, var=0.5, len_scale=2)

###############################################################################
# Generate correlated secondary data

np.random.seed(42)
sec_pos = np.linspace(0, 15, 51)
sec_val = 0.7 * np.interp(sec_pos, cond_pos, cond_val) + 0.3 * np.sin(sec_pos / 3)
sec_grid = np.interp(gridx, sec_pos, sec_val)
sec_at_primary = np.interp(cond_pos, sec_pos, sec_val)

###############################################################################
# Simple Kriging and Intrinsic Collocated Cokriging

sk = krige.Simple(model, cond_pos=cond_pos, cond_val=cond_val, mean=1)
sk_field, sk_var = sk(gridx, return_var=True)

cross_corr = np.corrcoef(cond_val, sec_at_primary)[0, 1]
icck = IntrinsicCollocated(
    model,
    cond_pos=cond_pos,
    cond_val=cond_val,
    secondary_cond_pos=cond_pos,
    secondary_cond_val=sec_at_primary,
    cross_corr=cross_corr,
    secondary_var=np.var(sec_val),
    mean=1,
    secondary_mean=np.mean(sec_val),
)
icck_field, icck_var = icck(gridx, secondary_data=sec_grid, return_var=True)

###############################################################################

plt.plot(gridx, sk_field, label="Simple Kriging")
plt.plot(gridx, icck_field, label="Intrinsic Collocated Cokriging")
plt.scatter(cond_pos, cond_val, color="k", zorder=10, label="Conditions")
plt.legend()
plt.show()
