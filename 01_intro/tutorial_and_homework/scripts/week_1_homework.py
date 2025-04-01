# %% [markdown]
#
# Use the **Pandas cheatsheet** [here](https://github.com/adrien-perello/Computer-Science-Crash-Course/blob/main/02_scientific-computing-libraries/03_pandas_cheatsheet.ipynb) to complete the tasks below
#

# %% [markdown]
# # Load libraries
#

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd

# %%
# Set the root directory to be able to access other folders in the project

CURR_DIR = os.getcwd()  # current working directory
BASE_DIR = Path(CURR_DIR).parent  # parent directory
print(BASE_DIR)

# %% [markdown]
# # Import a file and explore the data
#

# %%
# Import the "commodities.csv" file in the data folder
# (look for the "Read and write data from external files" section
# in the pandas cheatsheet)

# ! ANSWER
file_path = BASE_DIR / "data" / "raw" / "commodities.csv"
commodities = pd.read_csv(file_path)
commodities

# %%
# Import the commodities sheet from the 'MFA_II_pandas_exercise.xlsx' file

# ! ANSWER
file_path = BASE_DIR / "data" / "raw" / "MFA_II_pandas_exercise.xlsx"
commodities = pd.read_excel(file_path, sheet_name="commodities")
commodities

# %% [markdown]
# **Follow up questions:**
#
# What happens if you load the Excel file without specifying a sheet?
#
# 1. How does pandas handle multi-sheet Excel files by default?
# 2. What type of data structure does `pd.read_excel()` return in this case?
# 3. Can you still access the data from individual sheets after loading the file without specifying a sheet?
#
# Same questions as above, but with `sheet_name = None`
#

# %% [markdown]
# **ANSWERS**
#

# %% [markdown]
# ...
#

# %%
# get an overview of the data by printing
# 1) technical information about the dataframe
# 2) the number of the rows and columns
# 3) the first and/or last 3 rows of the dataframe
# 4) the name of the columns and of the index

# ! ANSWERS

# %% [markdown]
# # Select columns
#

# %%
# Display the "year" column
# 1) using []
# 2) using .loc[]
# 3) using .iloc[]

# ! ANSWERS

# %% [markdown]
# **Follow up questions:**
#
# 4. How do `[]`, `.loc[]`, and `.iloc[]` differ in their approach to selecting columns?
#
#    - What type of inputs do each of them accept?
#    - How does their output differ (if at all) in this case?
#
# 5. What happens if the column name changes or is misspelled in the dataset? What error is raised?
#
# 6. What happens if you try to use `.iloc[]` with a column index that doesn't exist? What error is raised? What about with negative numbers (e.g. -1 or -2)?
#
# 7. What are the advantages of using `.loc[]` vs `.iloc[]`? In what situations would one method be preferable over the others?
#

# %% [markdown]
# **ANSWERS**
#

# %% [markdown]
# ...
#

# %%
# Display multiple columns at once (e.g. "candies" and "patatoes")
# using each of the 3 methods

# ! ANSWERS

# %% [markdown]
# # Reindex
#

# %%
# Set "year" as Index
# make sure the change is permanent (2 options)

# ANSWER

# %% [markdown]
# **Follow up questions:**
#
# 8. what data structure does each option returns?
# 9. How can you reset the index back to the default integer index?
# 10. What happens if you try to set an index on a column that doesn’t exist?
# 11. Can you set multiple columns as an index? If so, how?
#

# %% [markdown]
# **ANSWERS**
#

# %% [markdown]
# ...
#

# %% [markdown]
# # Slice the data
#

# %%
# Display the number of each commodity for the year 2000
# in a single line of code.
# Provide solutions using both .loc[] and .iloc[].

# ! ANSWER

# %%
# Select the 'patatoes' and 'carrots' columns together
# for the year 2000, in a single line of code.
# Provide solutions using both .loc[] and .iloc[].

# ! ANSWER

# %%
# Do the same (show patatoes and carrots together)
# but for the year 2000 AND 2010.
# Provide solutions using both .loc[] and .iloc[].

# ! ANSWER

# %% [markdown]
# **Follow up questions:**
#
# 12. How would you select data for all years greater than 1995?
# 13. How would you select data for all years greater than 1995 and less than 2005?
# 14. How would you select data related either to `Alex` or `Lisa`?
# 15. load `commodity_alt.csv`, set `year` as index and try to access the data for the year 2000. Do you encounter any problem? If so, why is that?
#
# For the first two questions, try both using `:` and **boolean mask**.
#

# %% [markdown]
# **ANSWERS**
#

# %% [markdown]
# ...
#

# %% [markdown]
# # Clean the data
#

# %%
# rename the column that has a typo

# ! ANSWER

# %% [markdown]
# **Follow up questions:**
#
# 16. What happens if you try to rename a column that doesn’t exist in the DataFrame?
# 17. Is renaming case-sensitive? What if you accidentally change "Patatoes" to "Potatoes" but the original column is lowercase?
#

# %% [markdown]
# **ANSWERS**
#
# ...
#

# %%
# Remove the rows that are empty

# ! ANSWER

# %% [markdown]
# **Follow up questions:**
#
# 18. What happens if there are only some missing values in a row? Does .dropna() remove the whole row?
# 19. How could you replace missing values instead of removing them?
#

# %% [markdown]
# **ANSWERS**
#

# %% [markdown]
# ...
#

# %% [markdown]
# # Plot data
#

# %%
# Load the "wasteStatistics" sheet of the "MFA_II_pandas_exercise.xlsx" file,

# ! ANSWER

# %%
# Plot residual waste over the years (x-axis should be the year)

# ! ANSWER

# %% [markdown]
# **Follow up questions:**
#
# 20. How can you customize the plot to add a title and axis labels? Change the plot size?
# 21. Assuming the unit is kilograms, how would you display the results in tonnes?
#

# %% [markdown]
# **ANSWERS**
#

# %% [markdown]
# ...
#

# %%
# Plot all types of waste stacked over the years in one single plot

# ! ANSWER

# %% [markdown]
# # Mean, sum, maximum and minimum
#

# %%
# Calculate the mean, sum, minimum, and maximum for each year
# without using a loop. Save the results in a separate column.

# ! ANSWER

# %%
# still without using a loop, perform the following calculations:
# 1) the cumulative sum of organic waste over the years.
# 2) the year-over-year difference in chemical waste
# (i.e., the difference between each year and the previous one)
# save the results in separate columns

# ! ANSWER

# %% [markdown]
# **Follow up questions:**
#
# 22. What would happen if the dataset was not sorted by year before performing these operations?
# 23. What happens to the first row when calculating differences on a column with pandas? Why?
# 24. How can you modify the result so it shows 0 instead?
# 25. In some cases, we may want the first difference to be calculated from a specific **non-zero** starting value. How can we achieve this using numpy? How would you save the result (a numpy array) in a new column?
#

# %% [markdown]
# **ANSWERS**
#

# %% [markdown]
# ...
#

# %% [markdown]
# # Export file
#

# %%
# save the results as a csv file

# ! ANSWER

# %% [markdown]
# # Structure your project
#

# %% [markdown]
# We recommend following this folder structure, especially for larger projects like your thesis.
#
# For the final assignment, you only need a **single Python script** that implements all three steps: data cleaning, analysis, and visualization.
#
# ```bash
# project_name/
# │── data/
# │   ├── raw/          # Original data as downloaded (never modified)
# │   ├── interim/      # Data after initial cleaning/preprocessing
# │   ├── processed/    # Final, analysis-ready datasets
# │
# │── docs/
# │   ├── assignment.pdf  # Assignment details
# │   ├── references/     # Supporting materials (papers, articles, etc.)
# │
# │── img/               # Any images used (logos, schematics, etc.)
# │
# │── notebooks/
# │   ├── 00_exploratory.ipynb    # Initial exploration
# │   ├── 01_cleaning.ipynb       # Data cleaning & transformation
# │   ├── 02_analysis.ipynb       # Core computations & analysis
# │   ├── 03_visualization.ipynb  # Final graphs & insights
# │
# │── scripts/
# │   ├── 01_cleaning.py        # code for data cleaning & transformation
# │   ├── 02_analysis.py        # Core computations & analysis
# │   ├── 03_visualization.py   # Final plotting & insights
# │
# │── reports/
# │   ├── figures/       # Saved plots and figures
# │   ├── final_report_or_poster.pdf  # Final report or presentation
# │
# │── environment.yml    # Python dependencies (for Anaconda / Mamba)
# │── requirements.txt   # Python dependencies (for Pip / Virtualenv)
# ```
#

# %% [markdown]
# **Note**: to export a jupyter notebook file as a python script, click on `Save and Export Notebook As` > `Executable Script`.
# Don't forget to test the script to make sure everything works as intented.
#
# !["export as .py"](../img/export_as_python_script.png)
#

# %% [markdown]
# # More information and tips
#

# %% [markdown]
# The internet is your best friend for all Python-related questions! :smiley: You’ll find explanations, examples, and solutions for functions, errors, and code snippets online.
#
# 💡 How to search effectively?
# Use Google to look up functions or code, for example: `pandas combine data frames`
#
# 💡 LLMs can also be helpful...
# ...but use them wisely—double-check answers with official documentation and make sure to always understand how the code works
#
# 📚 Resources:
#
# - Forums: Especially check [Stackoverflow](https://stackoverflow.com/questions/tagged/pandas?tab=Frequent)
# - Pandas: check
#   - the [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/10min.html)
#   - the [Pandas API reference](https://pandas.pydata.org/docs/reference/frame.html),
#   - the [W3school doc](https://www.w3schools.com/python/pandas/default.asp)
#   - this [github repo](https://github.com/adrien-perello/Computer-Science-Crash-Course/blob/main/02_scientific-computing-libraries/03_pandas_cheatsheet.ipynb)
# - Numpy:
#   - the [Numpy User Guide](https://numpy.org/doc/stable/user/basics.html),
#   - the [Numpy API reference](https://numpy.org/doc/stable/reference/module_structure.html),
#   - the [W3school doc](https://www.w3schools.com/python/numpy/default.asp)
#   - this [github repo](https://github.com/adrien-perello/Computer-Science-Crash-Course/blob/main/02_scientific-computing-libraries/02_numpy-cheatsheet.ipynb)
# - Need a python refresher?
#   - check the [official Python doc](https://docs.python.org/3/tutorial/)
#   - the [W3school](https://www.w3schools.com/python/default.asp)
#   - this [github repo](https://github.com/adrien-perello/Computer-Science-Crash-Course/tree/main/01_introduction-to-python)
#
