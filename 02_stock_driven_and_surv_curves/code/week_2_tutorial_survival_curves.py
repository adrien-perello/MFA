# %% [markdown]
# # Load libraries
#

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import geom, lognorm, norm, uniform, weibull_min

# %%
# Set the root directory to be able to access other folders in the project

CURR_DIR = os.getcwd()  # current working directory
BASE_DIR = Path(CURR_DIR).parent  # parent directory
print(BASE_DIR)

# %% [markdown]
# # Import data
#

# %%
# Load input data, inflow-driven model:
# and check your data

file_path = BASE_DIR / "data_input" / "MFA_II_tutorial_II.xlsx"
data = pd.read_excel(file_path, sheet_name="inflow_driven")
data.info()

# %%
# set the index to year
data = data.set_index(["year"])

years = data.index
end_year = years[-1]
print(f"end_year = {end_year}")

data

# %%
step_max = data.shape[0]
timesteps = np.arange(0, step_max)
timesteps

# %% [markdown]
# # Create a survival curve
#
# (if one wasn't supplied as input data)
#

# %%
# Fixed lifetime survival curve

fixed_lifetime = 40
survival_curve = np.ones_like(timesteps)
survival_curve[fixed_lifetime:] = 0
plt.plot(survival_curve)
plt.show()

# %%
# Uniform distribution
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform

uniform_dist = uniform(
    loc=10,  # shifts the curve along the x-axis (starting point)
    scale=20,  # controls the width (ending point)
)
survival_curve = uniform_dist.sf(timesteps)  # sf = survival function
plt.plot(survival_curve)
plt.show()

# %%
# Geometric distribution
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html#scipy.stats.geom

geom_dist = geom(
    p=0.05,  # controls the depreciation rate
    loc=0,  # shifts the curve along the x-axis (starting point)
)
survival_curve = geom_dist.sf(timesteps)  # sf = survival function
plt.plot(survival_curve)
plt.show()

# %%
# Normal distribution
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm

norm_dist = norm(
    loc=30,  # shifts the center of the curve (mean point)
    scale=10,  # Controls the spread of the curve (standard deviation)
)
survival_curve = norm_dist.sf(timesteps)  # sf = survival function
plt.plot(survival_curve)
plt.show()

# %%
# Weibull distribution
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html

weibull_dist = weibull_min(
    c=2,  # controls the shape of the curve (skewness)
    loc=0,  # shifts the curve along the x-axis (starting point)
    scale=30,  # Stretches or compresses the curve along the x-axis (spread)
)
survival_curve = weibull_dist.sf(timesteps)  # sf = survival function
plt.plot(survival_curve)
plt.show()

# %%
# Lognormal distribution
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm

lognorm_dist = lognorm(
    s=0.5,  # controls the shape of the curve (skewness)
    loc=0,  # shifts the curve along the x-axis (starting point)
    scale=10,  # controls the stretch of the distribution
)
survival_curve = lognorm_dist.sf(timesteps)  # sf = survival function
plt.plot(survival_curve)
plt.show()

# %% [markdown]
# # Create a survival curve matrix
#

# %%
# create survival curve matrix with placeholder zeros
survival_curve_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps, dtype=float)

# %% [markdown]
# ![filling_survival_curve_matrix](../../img/filling_survival_curve_matrix.png)
#

# %%
# populate the survival curve matrix with shifted curves, column by column using slices
for step in timesteps:
    # at each iteration, we take 1 year less of the survival curve
    last_idx = step_max - step
    values = survival_curve[0:last_idx]
    # and we assign the sliced values to the sliced matrix:
    # --> rows: from step to step_max
    # --> columns: only the current step
    survival_curve_matrix.loc[step:step_max, step] = values

survival_curve_matrix

# %%
# visualize the survival curve matrix with a heatmap
sns.heatmap(survival_curve_matrix, annot=False)

# %% [markdown]
# # Going further
#

# %%
# Instead of a numpy array, we store the survival curve
# as a pandas Series with the appropriate index (= years)
sf = pd.Series(survival_curve, index=years)
sf

# %%
# create survival curve matrix with placeholder zeros (same as before)
survival_curve_matrix2 = pd.DataFrame(0, index=years, columns=years, dtype=float)

# populate the survival curve matrix with shifted curves, column by column using slices
# ! This time we use the years as index instead of the timesteps
for counter, year in enumerate(years):
    # at each iteration, we take 1 year less of the survival curve
    last_idx = end_year - counter
    values = sf.loc[:last_idx].values
    # and we assign the sliced values to the sliced matrix:
    # --> rows: from current year to the end year
    # --> columns: only the current year
    survival_curve_matrix2.loc[year:end_year, year] = values

# notice the names of the columns and rows
# are now years instead of timesteps
survival_curve_matrix2

# %%
# visualize the cohort matrix with a heatmap
# notice the names of the columns and rows
# are now years instead of timesteps
sns.heatmap(survival_curve_matrix2, annot=False)

# %% [markdown]
# # More information and tips
#

# %% [markdown]
# - [Statistical functions in scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
# - the `enumerate()` function on [W3school](https://www.w3schools.com/python/ref_func_enumerate.asp) or [Programiz](https://www.programiz.com/python-programming/methods/built-in/enumerate)
#
