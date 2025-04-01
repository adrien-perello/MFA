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

file_path = BASE_DIR / "data" / "raw" / "MFA_II_tutorial_II.xlsx"
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
# --> see `week_2_tutorial_survival_curves.ipynb` notebook first
#

# %%
# Weibull distribution
weibull_dist = weibull_min(
    c=2,  # controls the shape of the curve (skewness)
    loc=0,  # shifts the curve along the x-axis (starting point)
    scale=30,  # Stretches or compresses the curve along the x-axis (spread)
)
survival_curve = weibull_dist.sf(timesteps)  # sf = survival function

plt.plot(survival_curve)
plt.show()

# %% [markdown]
# # Create a survival curve matrix
#

# %% [markdown]
# --> see `week_2_tutorial_survival_curves.ipynb` notebook first
#

# %%
# create survival curve matrix with placeholder zeros
survival_curve_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps, dtype=float)

# %% [markdown]
# ![filling_survival_curve_matrix](../img/filling_survival_curve_matrix.png)
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

# %% [markdown]
# # Flow driven model
#

# %%
# This is our model input
inflows = data["inflow"]
inflows

# %%
# create survival matrix with placeholder zeros that we will populate
cohort = pd.DataFrame(0, index=timesteps, columns=timesteps, dtype=float)

# %% [markdown]
# ![flow_driven_equations](../img/flow_driven_equations.png)
#
# ![filling_flow_driven_cohort](../img/filling_flow_driven_cohort.png)
#

# %%
# multiply the inflow with the shifted curves to get the cohorts' behavior over time
for time in timesteps:
    # we multiply the vector 'survival_curve_matrix.loc[:, time]'
    # with the scalar 'inflows.iloc[time]'
    cohort.loc[:, time] = survival_curve_matrix.loc[:, time] * inflows.iloc[time]

cohort

# %%
# set index and columns to years instead of timesteps
cohort.index = years
cohort.columns = years

# and visualize the cohort matrix with a heatmap
sns.heatmap(cohort, annot=False)

# %%
# calculate flows & stocks using the cohort
stock = cohort.sum(axis=1)
nas = np.diff(stock, prepend=0)  # prepending 0 assumes no initial stock
outflows = inflows - nas

# %% [markdown]
# # Visualize the results
#

# %%
data["stock"] = stock
data["outflow"] = outflows
data["nas"] = nas

data

# %%
# Visualize on the same plot
data.plot()

# %%
# Zooming in the flows and net addition to stock
data[["inflow", "outflow", "nas"]].plot()

# %%
# Zooming in the stocks
data["stock"].plot()

# %% [markdown]
# # Export output data to Excel
#

# %%
# Save the data to an Excel file
# (you may need to create the folder if it doesn't exist)
file_path = BASE_DIR / "data" / "processed" / "week_2_tutorial_myname.xlsx"
data.to_excel(file_path, sheet_name="flow_driven")

# %%
# But we also want to save the cohort data in the same excel file
# without overwriting the file.
# To do that, we open an Excel file in append mode ('a')
# https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html

with pd.ExcelWriter(file_path, mode="a") as writer:
    cohort.to_excel(writer, sheet_name="cohort_flow_driven")

# %% [markdown]
# # Going further
#
# --> We saw in `week_2_tutorial_survival_curves.ipynb` notebook that we could also define the `survival_curve` and the `survival_curve_matrix` as follows:
#

# %%
# Instead of a numpy array, we store the survival curve
# as a pandas Series with the appropriate index (= years)
sf = pd.Series(survival_curve, index=years)

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

# %% [markdown]
# Now let's utilize numpy and pandas capacities to **optimize** our flow driven model
#

# %% [markdown]
# There are different types of matrix multiplications:
#
# - the **dot product** (using the `np.dot()` function or the `@` operator)?
# - the **element-wise matrix multiplication** (`np.multiply()` function or `*` operator)
#
# We already saw the element-wise matrix multiplication when computing the cohort matrix.
# Let's dive further in what it does:
#
# $$
# \begin{bmatrix}
# a_{11} & a_{12} & a_{13} \\
# a_{21} & a_{22} & a_{23} \\
# a_{31} & a_{32} & a_{33} \\
# \end{bmatrix}
# *
# \begin{bmatrix}
# f_1 \\
# f_2 \\
# f_3
# \end{bmatrix}
# =
# \begin{bmatrix}
# a_{11} f_1  & a_{12} f_1 & a_{13} f_1 \\
# a_{21} f_2  & a_{22} f_2 & a_{23} f_2 \\
# a_{31} f_3  & a_{32} f_3 & a_{33} f_3 \\
# \end{bmatrix}
# $$
#
# Now assuming that
#
# - the matrix $\mathbf{A}$ represents the `survival_curve_matrix`
# - and the vector $\mathbf{f}$ represents the `inflows`,
#
# we get:
#
# $$
# \begin{bmatrix}
# a_{1} & 0 & 0 \\
# a_{2} & a_{1} & 0 \\
# a_{3} & a_{2} & a_{1} \\
# \end{bmatrix}
# *
# \begin{bmatrix}
# f_1 \\
# f_2 \\
# f_3
# \end{bmatrix}
# =
# \begin{bmatrix}
# a_{1} f_1 & 0 & 0 \\
# a_{2} f_1 & a_{1} f_2 & 0 \\
# a_{3} f_1 & a_{2} f_2 & a_{1} f_3  \\
# \end{bmatrix}
# $$
#
# You may recognize the `cohort` matrix $\mathbf{C}$ as the result.
#
# In other words, the **element-wise matrix multiplication** saves us from having to do a `for loop`
#

# %%
# no need to do a for loop with the element wise multiplication
# and notice that this time, we don't need to manually set
# the rows and columns to the years
cohort2 = survival_curve_matrix2 * inflows
cohort2

# %%
# Check that they are indeed the same
# always use np.allclose() instead of == to compare floats
np.allclose(cohort2, cohort)

# %%
# visualize the cohort matrix with a heatmap
sns.heatmap(cohort2, annot=False)

# %% [markdown]
# # More information and tips
#

# %% [markdown]
# - [Statistical functions in scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
# - the `enumerate()` function on [W3school](https://www.w3schools.com/python/ref_func_enumerate.asp) or [Programiz](https://www.programiz.com/python-programming/methods/built-in/enumerate)
# - [Pandas excel writer](https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html)
# - [Element wise multiplication](https://www.sharpsightlabs.com/blog/numpy-multiply/)
# - Difference between `np.dot()`, `np.multiply()` and `np.matmul()` on [codeacademy](https://www.codecademy.com/article/numpy-matrix-multiplication-a-beginners-guide) or [digitalocean](https://www.digitalocean.com/community/tutorials/numpy-matrix-multiplication)
#
