r"""
Simple Collocated Cokriging
----------------------------

Simple collocated cokriging uses secondary data at the estimation location
to improve the primary variable estimate.

This uses the Markov Model I (MM1) approach:

.. math:: C_{YZ}(h) = \rho_{YZ}(0) \cdot C_Z(h)

Example
^^^^^^^

Here we compare Simple Kriging with Simple Collocated Cokriging.
"""

import matplotlib.pyplot as plt
import numpy as np

from gstools import Gaussian, krige
from gstools.cokriging import SimpleCollocated

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
# Simple Kriging and Simple Collocated Cokriging

sk = krige.Simple(model, cond_pos=cond_pos, cond_val=cond_val, mean=1)
sk_field, sk_var = sk(gridx, return_var=True)

cross_corr = np.corrcoef(cond_val, sec_at_primary)[0, 1]
scck = SimpleCollocated(
    model,
    cond_pos=cond_pos,
    cond_val=cond_val,
    cross_corr=cross_corr,
    secondary_var=np.var(sec_val),
    mean=1,
    secondary_mean=np.mean(sec_val),
)
scck_field, scck_var = scck(gridx, secondary_data=sec_grid, return_var=True)

###############################################################################

plt.plot(gridx, sk_field, label="Simple Kriging")
plt.plot(gridx, scck_field, label="Simple Collocated Cokriging")
plt.scatter(cond_pos, cond_val, color="k", zorder=10, label="Conditions")
plt.legend()
plt.show()
