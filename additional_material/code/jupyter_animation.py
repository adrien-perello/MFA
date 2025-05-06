# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import abc
from typing import Any, Dict, Optional, Tuple, Type

import ipywidgets as widgets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display

# ---------------------------------------------------------------------------
# 2. Type Definitions/Aliases
# ---------------------------------------------------------------------------
# Using basic types for broad compatibility.
# Consider using numpy.typing.NDArray for more specific NumPy array typing.
NDArray = np.ndarray
AnimationData = Dict[str, Any]  # Represents the prepared data structure for plotting

# ---------------------------------------------------------------------------
# 3. Core Implementation Functions
#    (These perform the actual data prep and plotting for each model type)
#    REPLACE PLACEHOLDER LOGIC WITH YOUR ACTUAL IMPLEMENTATIONS
# ---------------------------------------------------------------------------

# --- Flow-Driven Model Functions ---


def get_flow_driven_animation_data(
    inflow: NDArray,
    curve_surv_matrix: NDArray,
    cohort_surv_matrix: NDArray,
    timesteps: range,
    all_steps: Optional[bool] = False,
    **kwargs: Any,
) -> AnimationData:
    """Prepares data structures required for animating the flow-driven model calculation.

    For each timestep, it isolates the relevant inflow, survival matrix elements,
    and the resulting calculated stock value corresponding to that step, storing them
    in dictionaries keyed by time. It also generates the LaTeX equation string
    representing the stock calculation at each step.

    Args:
        inflow (1D array): The input inflow at each timestep (model input).
        curve_surv_matrix (2D array): Survival probabilities, where
                            `curve_surv_matrix[t, cohort_t]` is the survival
                            probability of `cohort_t` at observation time `t`.
        cohort_surv_matrix (2D array): where `cohort_surv_matrix[t, cohort_t]`
                            gives the surviving amount *from* the `cohort_t` inflow
                            *at* observation time `t`.
        timesteps: A range object representing the time indices (0 to N-1).

    Returns:
        A dictionary containing data structured for animation:
        - "inflows": Dict mapping time -> input inflow value at that time step.
        - "curve_surv_matrices": Dict mapping time -> relevant survival curve column.
        - "cohort_surv_matrices": Dict mapping time -> relevant cohort survival matrix row.
        - "time_max": The total number of timesteps (N).
        - "vmax_inflow": Maximum inflow value for colorbar scaling.
        - "inflow_plot_shape": Shape tuple for plotting inflow (1, N).
        - "curve_plot_shape": Shape tuple for plotting curve matrix (N, N).
        - "cohort_plot_shape": Shape tuple for plotting cohort matrix (N, N).
    """
    time_max = len(timesteps)
    animation_inflow = dict()
    animation_curve_surv = dict()
    animation_cohort_surv = dict()

    # Pre-fill NaN arrays for efficient updates
    nan_col_vector = np.full((time_max, 1), np.nan, dtype=float)
    nan_matrix = np.full((time_max, time_max), np.nan, dtype=float)

    # If all_steps is True, we show current steps + all previous steps
    if all_steps:

        for time in timesteps:
            if time == 0:
                # Initialize the first time step with NaN
                animation_inflow[time] = nan_col_vector.copy()
                animation_curve_surv[time] = nan_matrix.copy()
                animation_cohort_surv[time] = nan_matrix.copy()
            else:
                # Copy the previous time step's data
                animation_inflow[time] = animation_inflow[time - 1].copy()
                animation_curve_surv[time] = animation_curve_surv[time - 1].copy()
                animation_cohort_surv[time] = animation_cohort_surv[time - 1].copy()

            # Highlight the values calculated at 'time'
            animation_inflow[time][time, 0] = inflow[time]
            animation_curve_surv[time][time:, time] = curve_surv_matrix[time:, time]
            animation_cohort_surv[time][time:, time] = cohort_surv_matrix[time:, time]

    # If all_steps is False, we only show current steps
    else:
        for time in timesteps:
            # Initialize with NaN
            animation_inflow[time] = nan_col_vector.copy()
            animation_curve_surv[time] = nan_matrix.copy()
            animation_cohort_surv[time] = nan_matrix.copy()
            # Highlight the values calculated at 'time'
            animation_inflow[time][time, 0] = inflow[time]
            animation_curve_surv[time][time:, time] = curve_surv_matrix[time:, time]
            animation_cohort_surv[time][time:, time] = cohort_surv_matrix[time:, time]

    animation_data = {
        "inflows": animation_inflow,
        "curve_surv_matrices": animation_curve_surv,
        "cohort_surv_matrices": animation_cohort_surv,
        "time_max": time_max,
        "vmax_inflow": np.nanmax(inflow),
        "inflow_plot_shape": (1, time_max),
        "curve_plot_shape": (time_max, time_max),
        "cohort_plot_shape": (time_max, time_max),
    }
    return animation_data


def flow_driven_animation(animation_data: AnimationData, figsize: Tuple[float, float]) -> plt.Figure:
    """Generates an interactive animation visualizing the flow-driven model calculation.

    Displays heatmaps for inflow, survival curve matrix, and cohort survival matrix
    components, along with the step-by-step calculation formula.
    Uses an ipywidgets slider to control the current timestep being visualized.

    Args:
        animation_data (dict): The dictionary generated by `get_flow_driven_animation_data`.
        figsize (tuple, optional): The size of the matplotlib figure.

    Returns:
        The matplotlib Figure object containing the animation setup.
        The interactive slider is displayed separately using `IPython.display`.
    """
    # --- Create the figure and axes using GridSpec ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 15], width_ratios=[1, 1])

    # Assign axes to grid locations
    ax_inflow = fig.add_subplot(gs[0, 0])  # Top-left: Inflow value display
    ax_time_display = fig.add_subplot(gs[0, 1])  # Top-right: Time step display
    ax_curve = fig.add_subplot(gs[1, 0])  # Mid-left: Curve Surv Matrix
    ax_cohort = fig.add_subplot(gs[1, 1])  # Bottom-left: Cohort Surv Matrix

    # Store axes in a dictionary for easier access
    axes = {
        "inflow": ax_inflow,
        "time_display": ax_time_display,
        "curve": ax_curve,
        "cohort": ax_cohort,
    }

    # Unpack animation data (NumPy arrays with NaNs)
    inflows = animation_data["inflows"]  # This now highlights contributing inflows
    curve_surv_matrices = animation_data["curve_surv_matrices"]
    cohort_surv_matrices = animation_data["cohort_surv_matrices"]
    time_max = animation_data["time_max"]
    vmax_inflow = animation_data["vmax_inflow"]  # Max of the original inflow vector

    # Shapes for placeholders (ensures correct heatmap dimensions initially)
    inflow_plot_shape = animation_data["inflow_plot_shape"]  # (1, N) - No longer plotting inflow heatmap
    curve_plot_shape = animation_data["curve_plot_shape"]  # (N, N)
    cohort_plot_shape = animation_data["cohort_plot_shape"]  # (N, N)

    # Initialize placeholders with ZEROS (heatmap handles NaN display later)
    placeholder_curve = np.zeros(curve_plot_shape)
    placeholder_cohort = np.zeros(cohort_plot_shape)
    placeholder_inflow = np.zeros(inflow_plot_shape)

    # --- Create the heatmaps ONCE (critical for performance) ---
    # Common heatmap settings
    heatmap_settings = {"annot": False, "vmin": 0, "cbar": True, "square": True}
    # Specific settings for matrices (thinner colorbar)
    heatmap_settings_matrix = {**heatmap_settings, "cbar_kws": {"shrink": 0.5, "aspect": 30}}
    # Specific settings for stock/inflow vectors (fatter colorbar)
    heatmap_settings_vector = {**heatmap_settings, "cbar_kws": {"shrink": 0.8, "aspect": 3}}  # For stock row

    # Create heatmaps using placeholders and store their collection objects
    sns.heatmap(placeholder_inflow, vmax=vmax_inflow, ax=axes["inflow"], cmap="copper_r", **heatmap_settings_vector)
    sns.heatmap(placeholder_curve, vmax=1, ax=axes["curve"], cmap="bone_r", **heatmap_settings_matrix)
    sns.heatmap(placeholder_cohort, vmax=vmax_inflow, ax=axes["cohort"], cmap="copper_r", **heatmap_settings_matrix)

    # Store collection objects
    collections = {
        "inflow": axes["inflow"].collections[0],
        "curve": axes["curve"].collections[0],
        "cohort": axes["cohort"].collections[0],
    }

    # --- Text Settings ---
    text_settings = {"horizontalalignment": "center", "verticalalignment": "center", "fontsize": 12}

    # --- Create Text objects ONCE ---
    # Turn off axis for the time display area
    axes["time_display"].set_axis_off()
    time_text_obj = axes["time_display"].text(
        0.5, 0.5, "Time step = 0", transform=axes["time_display"].transAxes, **text_settings
    )

    # --- Set static titles and axis properties ---
    axes["cohort"].set_title("Cohort Survival Matrix")
    axes["curve"].set_title("Survival Curve Matrix")
    axes["inflow"].set_yticks([])  # Not needed anymore
    axes["inflow"].set_ylabel("")

    # --- Define the update function ---
    def update_heatmap(time):
        """Updates the plot elements for the given timestep."""
        # --- Retrieve pre-calculated full NumPy arrays (with NaNs) ---
        inflow_val = inflows[time][time, 0]  # Get specific inflow at 'time' for display
        inflow_data_full = inflows[time]  # Shape (N, 1)
        curve_data_full = curve_surv_matrices[time]  # Shape (N, N)
        cohort_data_full = cohort_surv_matrices[time]  # Shape (N, N)

        # --- Prepare data for set_array ---
        inflow_plot_data = inflow_data_full.T.flatten()  # (N,1) -> (1,N) -> flatten
        curve_plot_data = curve_data_full.flatten()
        cohort_plot_data = cohort_data_full.flatten()

        # --- Update Heatmap Data ---
        collections["inflow"].set_array(inflow_plot_data)
        collections["curve"].set_array(curve_plot_data)
        collections["cohort"].set_array(cohort_plot_data)

        # --- Update dynamic titles/text ---
        axes["inflow"].set_title(f"Inflow = {inflow_val}")  # Format value

        # --- Update Equations ---
        time_text_obj.set_text(f"Time step = {time}")

        fig.canvas.draw_idle()

    # --- Create the slider ---
    step_slider = widgets.IntSlider(
        min=0, max=time_max - 1, value=0, description="Time step", continuous_update=False, layout={"width": "85%"}
    )

    # Connect slider
    widgets.interactive_output(update_heatmap, {"time": step_slider})

    # --- Display ---
    display(step_slider)
    fig.tight_layout(pad=1.5)

    # Apply canvas settings from your code
    # Note: These might depend on the specific backend/version (like ipympl)
    # and might potentially conflict with rcParams settings if you used them.
    try:
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_position = "right"
    except AttributeError:
        print("Warning: Canvas settings might not be available for this backend.")

    update_heatmap(0)

    return fig


# --- Stock-Driven Model Functions ---
def get_stock_driven_animation_data(
    stock: NDArray,
    inflow: NDArray,
    curve_surv_matrix: NDArray,
    cohort_surv_matrix: NDArray,
    timesteps: range,
    all_steps: Optional[bool] = False,
    **kwargs: Any,
) -> AnimationData:
    """Prepares data structures required for animating the stock-driven model calculation.

    For each timestep, it isolates the relevant stock, inflow, survival curve and cohort
    values corresponding to that step in the calculation process, storing them in
    dictionaries keyed by time. It also generates the LaTeX equation string
    representing the calculation at each step.

    Args:
        stock (1D array): represents the stock level at each timestep (model input).
        inflow (1D array): represents the calculated inflow at each timestep.
        curve_surv_matrix (2D array): where `curve_surv_matrix[t, cohort_t]`
                            gives the survival probability of the `cohort_t`
                            at observation time `t`. Assumed to be lower triangular
                            or appropriately masked.
        cohort_surv_matrix (2D array): where `cohort_surv_matrix[t, cohort_t]`
                            gives the surviving amount *from* the `cohort_t` inflow
                            *at* observation time `t`.
        timesteps: A range object representing the time indices.

    Returns:
        A dictionary containing data structured for animation:
        - "stocks": Dict mapping time -> stock value at that time step (as Nx1 NaN array).
        - "inflows": Dict mapping time -> inflow value at that time step (as Nx1 NaN array).
        - "curve_surv_matrices": Dict mapping time -> relevant column of survival curve matrix.
        - "cohort_surv_matrices": Dict mapping time -> relevant row elements for summation.
        - "cohort_surv_matrices_2": Dict mapping time -> relevant column elements for summation.
        - "time_max": The total number of timesteps (N).
        - "equations": List of LaTeX strings representing the calculation at each step.
        - "vmax_stock": Maximum stock value for colorbar scaling.
        - "vmax_inflow": Maximum inflow value for colorbar scaling.
        - "stock_plot_shape": Shape tuple for plotting stock (1, N).
        - "inflow_plot_shape": Shape tuple for plotting inflow (1, N).
        - "curve_plot_shape": Shape tuple for plotting curve matrix (N, N).
        - "cohort_plot_shape": Shape tuple for plotting cohort matrix (N, N).
    """
    time_max = len(timesteps)
    animation_stock = dict()
    animation_inflow = dict()
    animation_curve_surv = dict()
    animation_cohort_surv_1 = dict()
    animation_cohort_surv_2 = dict()
    eq_sum = np.zeros(time_max, dtype=float)
    equations = []

    # Pre-fill NaN arrays for efficient updates
    nan_col_vector = np.full((time_max, 1), np.nan, dtype=float)
    nan_matrix = np.full((time_max, time_max), np.nan, dtype=float)

    # If all_steps is True, we show current steps + all previous steps
    if all_steps:
        for time in timesteps:
            if time == 0:
                # Initialize the first time step with NaN (besides for cohort_surv_1)
                animation_stock[time] = nan_col_vector.copy()
                animation_inflow[time] = nan_col_vector.copy()
                animation_curve_surv[time] = nan_matrix.copy()
                animation_cohort_surv_2[time] = nan_matrix.copy()
            else:
                # Copy the previous time step's data (besides for cohort_surv_1)
                animation_stock[time] = animation_stock[time - 1].copy()
                animation_inflow[time] = animation_inflow[time - 1].copy()
                animation_curve_surv[time] = animation_curve_surv[time - 1].copy()

                animation_cohort_surv_2[time] = animation_cohort_surv_2[time - 1].copy()

            # Highlight the values calculated at 'time'
            animation_stock[time][time, 0] = stock[time]
            animation_inflow[time][time, 0] = inflow[time]
            animation_curve_surv[time][time:, time] = curve_surv_matrix[time:, time]
            animation_cohort_surv_2[time][time:, time] = cohort_surv_matrix[time:, time]
            animation_cohort_surv_1[time] = nan_matrix.copy()
            animation_cohort_surv_1[time][time, :time] = cohort_surv_matrix[time, :time]

    # If all_steps is False, we only show current steps
    else:
        for time in timesteps:
            # Initialize with NaN
            animation_stock[time] = nan_col_vector.copy()
            animation_inflow[time] = nan_col_vector.copy()
            animation_curve_surv[time] = nan_matrix.copy()
            animation_cohort_surv_1[time] = nan_matrix.copy()
            animation_cohort_surv_2[time] = nan_matrix.copy()
            # Highlight the values calculated at 'time'
            animation_stock[time][time, 0] = stock[time]
            animation_inflow[time][time, 0] = inflow[time]
            animation_curve_surv[time][time:, time] = curve_surv_matrix[time:, time]
            animation_cohort_surv_1[time][time, :time] = cohort_surv_matrix[time, :time]
            animation_cohort_surv_2[time][time:, time] = cohort_surv_matrix[time:, time]

    for time in timesteps:
        # summation
        if time > 0:
            eq_sum[time] = cohort_surv_matrix[time, :time].sum()

        # --- Generate Equation String ---
        # Special case for the first timestep (no summation sigma sign)
        if time == 0:
            # Line 1 (LaTeX - real equation)
            line1 = r"$\dfrac{{ \mathrm{{stock}}(0) }}{{ \mathrm{{surv}}(0) }}$"
            # Line 2 (Numerical Step 1)
            line2 = rf"$= \dfrac{{ {stock[time]} }}{{ {curve_surv_matrix[time, time]} }}$"
            eq_str = line1 + "\n\n\n" + line2

        # Special case for the first timestep (still no summation sigma sign)
        elif time == 1:
            eq_sum[time] = cohort_surv_matrix[time, :time].sum()
            # Line 1 (LaTeX - real equation)
            line1 = r"$\dfrac{{ \mathrm{{stock}}(1) - \mathrm{{inflow}}(0) \times \mathrm{{surv}}(1) }}{{ \mathrm{{surv}}(0) }}$"
            # Line 2 (LaTeX - pandas equation)
            line2 = (
                rf"$= \dfrac{{ "
                rf"\mathrm{{stock}}({time}) - "
                rf"\mathrm{{ CohortSurvMatrix.loc[{time}, : {time - 1}].sum() }}"
                rf" }}{{ \mathrm{{surv}}(0) }}$"
            )
            # Line 3 (Numerical Step 1)
            line3 = rf"$= \dfrac{{ {stock[time]} - {eq_sum[time]} }}{{ {curve_surv_matrix[time, time]} }}$"
            eq_str = line1 + "\n\n\n" + line2 + "\n\n\n" + line3

        # General case for all other timesteps (with summation sigma sign)
        else:
            eq_sum[time] = cohort_surv_matrix[time, :time].sum()
            # Line 1 (LaTeX - real equation)
            line1 = (
                rf"$\dfrac{{ "
                rf"\mathrm{{stock}}({time}) - "
                rf"\sum_{{t=0}}^{{ {time-1} }} \left( \mathrm{{inflow}}(t) \times \mathrm{{surv}}({time}-t) \right)"
                rf" }}{{ \mathrm{{surv}}(0) }}$"
            )
            # Line 2 (LaTeX - pandas equation)
            line2 = (
                rf"$= \dfrac{{ "
                rf"\mathrm{{stock}}({time}) - "
                rf"\mathrm{{ CohortSurvMatrix.loc[{time}, : {time - 1}].sum() }}"
                rf" }}{{ \mathrm{{surv}}(0) }}$"
            )
            # Line 3 (Numerical Step 1)
            line3 = rf"$= \dfrac{{ {stock[time]} - {eq_sum[time]} }}{{ {curve_surv_matrix[time, time]} }}$"
            eq_str = line1 + "\n\n\n" + line2 + "\n\n\n" + line3

        # Line 4 (Final Result)
        line_numerial_res = rf"$= {inflow[time]}$"
        line_inflow = rf"$= \mathrm{{inflow}}({time})$"
        # Combine lines with newline characters
        eq_str = eq_str + "\n\n\n" + line_numerial_res + "\n\n\n" + line_inflow

        equations.append(eq_str)

    animation_data = {
        "stocks": animation_stock,
        "inflows": animation_inflow,
        "curve_surv_matrices": animation_curve_surv,
        "cohort_surv_matrices": animation_cohort_surv_1,  # Part 1 (at t-1)
        "cohort_surv_matrices_2": animation_cohort_surv_2,  # Part 2 (at t)
        "time_max": time_max,
        "equations": equations,
        "vmax_stock": np.nanmax(stock),  # Max values for colorbars
        "vmax_inflow": np.nanmax(inflow),  # Max values for colorbars
        "stock_plot_shape": (1, time_max),  # shapes placeholders for heatmap
        "inflow_plot_shape": (1, time_max),  # shapes placeholders for heatmap
        "curve_plot_shape": (time_max, time_max),  # shapes placeholders for heatmap
        "cohort_plot_shape": (time_max, time_max),  # shapes placeholders for heatmap
    }
    return animation_data


def stock_driven_animation(animation_data: AnimationData, figsize: Tuple[float, float], **kwargs: Any) -> plt.Figure:
    """Generates an interactive animation visualizing the stock-driven model calculation.

    Displays heatmaps for stock, inflow, survival curve matrix, and cohort survival
    matrix components, along with the step-by-step calculation formula.
    Uses an ipywidgets slider to control the current timestep being visualized.

    Args:
        animation_data (dict): The dictionary generated by `get_animation_data`.
        figsize (tuple, optional): The size of the matplotlib figure.

    Returns:
        The matplotlib Figure object containing the animation setup.
        The interactive slider is displayed separately using `IPython.display`.
    """
    # --- Create the figure and axes using GridSpec ---
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 15, 1, 15], width_ratios=[1, 1])

    # Assign axes to grid locations
    ax_stock = fig.add_subplot(gs[0, 0])
    ax_time_display = fig.add_subplot(gs[0, 1])
    ax_cohort1 = fig.add_subplot(gs[1, 0])
    ax_equation = fig.add_subplot(gs[1, 1])
    ax_inflow = fig.add_subplot(gs[2, 0])
    ax_curve = fig.add_subplot(gs[3, 0])
    ax_cohort2 = fig.add_subplot(gs[3, 1])

    # Store axes in a dictionary for easier access
    axes = {
        "stock": ax_stock,
        "time_display": ax_time_display,
        "cohort1": ax_cohort1,
        "equation": ax_equation,
        "inflow": ax_inflow,
        "curve": ax_curve,
        "cohort2": ax_cohort2,
    }

    # Unpack animation data (NumPy arrays with NaNs)
    stocks = animation_data["stocks"]
    inflows = animation_data["inflows"]
    curve_surv_matrices = animation_data["curve_surv_matrices"]
    cohort_surv_matrices = animation_data["cohort_surv_matrices"]  # Part 1 (at t-1)
    cohort_surv_matrices_2 = animation_data["cohort_surv_matrices_2"]  # Part 2 (at t)
    time_max = animation_data["time_max"]
    vmax_stock = animation_data["vmax_stock"]
    vmax_inflow = animation_data["vmax_inflow"]
    equations = animation_data["equations"]

    # Shapes for placeholders (ensures correct heatmap dimensions initially)
    stock_plot_shape = animation_data["stock_plot_shape"]  # = (1, N)
    inflow_plot_shape = animation_data["inflow_plot_shape"]  # = (1, N)
    curve_plot_shape = animation_data["curve_plot_shape"]  # = (N, N)
    cohort_plot_shape = animation_data["cohort_plot_shape"]  # = (N, N)

    # Initialize placeholders with ZEROS (heatmap handles NaN display later)
    placeholder_stock = np.zeros(stock_plot_shape)
    placeholder_inflow = np.zeros(inflow_plot_shape)
    placeholder_curve = np.zeros(curve_plot_shape)
    placeholder_cohort1 = np.zeros(cohort_plot_shape)
    placeholder_cohort2 = np.zeros(cohort_plot_shape)

    # --- Create the heatmaps ONCE (critical for performance) ---
    # Common heatmap settings
    heatmap_settings = {"annot": False, "vmin": 0, "cbar": True, "square": True}
    # Specific settings for matrices (thinner colorbar)
    heatmap_settings_matrix = {**heatmap_settings, "cbar_kws": {"shrink": 1, "aspect": 40}}
    # Specific settings for stock/inflow vectors (fatter colorbar)
    heatmap_settings_vector = {**heatmap_settings, "cbar_kws": {"shrink": 1, "aspect": 4}}

    # Create heatmaps using placeholders and store their collection objects
    sns.heatmap(placeholder_stock, vmax=vmax_stock, ax=axes["stock"], cmap="viridis", **heatmap_settings_vector)
    sns.heatmap(placeholder_inflow, vmax=vmax_inflow, ax=axes["inflow"], cmap="copper_r", **heatmap_settings_vector)
    sns.heatmap(placeholder_curve, vmax=1, ax=axes["curve"], cmap="bone_r", **heatmap_settings_matrix)
    sns.heatmap(placeholder_cohort1, vmax=vmax_inflow, ax=axes["cohort1"], cmap="copper_r", **heatmap_settings_matrix)
    sns.heatmap(placeholder_cohort2, vmax=vmax_inflow, ax=axes["cohort2"], cmap="copper_r", **heatmap_settings_matrix)

    # Store the collection objects for direct access
    collections = {
        "stock": axes["stock"].collections[0],
        "inflow": axes["inflow"].collections[0],
        "curve": axes["curve"].collections[0],
        "cohort1": axes["cohort1"].collections[0],
        "cohort2": axes["cohort2"].collections[0],
    }

    # --- Settings for Text Display ---
    text_settings = {
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "fontsize": 12,
    }
    equation_text_settings = {**text_settings, "fontsize": 10}

    # --- Create Text objects ONCE ---
    # Turn off axis for the time display area
    axes["time_display"].set_axis_off()
    axes["equation"].set_axis_off()

    time_text_obj = axes["time_display"].text(
        0.5, 0.5, "Time step = 0", transform=axes["time_display"].transAxes, **text_settings
    )
    eq_text_obj = axes["equation"].text(
        0.5, 0.5, equations[0], transform=axes["equation"].transAxes, **equation_text_settings
    )

    # --- Set static titles and axis properties ---
    axes["cohort1"].set_title(f"Cohort Survival Matrix\n(at previous Time step)")
    axes["curve"].set_title("Survival Curve Matrix")
    axes["stock"].set_yticks([])
    axes["stock"].set_ylabel("")  # Match original
    axes["inflow"].set_yticks([])
    axes["inflow"].set_ylabel("")  # Match original

    # --- Define the update function ---
    def update_heatmap(time):
        """Updates the plot elements for the given timestep."""
        # Retrieve pre-calculated full NumPy arrays (with NaNs)
        stock_data_full = stocks[time]  # Shape (N, 1)
        inflow_data_full = inflows[time]  # Shape (N, 1)
        curve_data_full = curve_surv_matrices[time]  # Shape (N, N)
        cohort1_data_full = cohort_surv_matrices[time]  # Shape (N, N)
        cohort2_data_full = cohort_surv_matrices_2[time]  # Shape (N, N)

        # --- Prepare data for set_array ---
        # Stock/Inflow are plotted as 1xN heatmaps, so transpose (N,1)->(1,N) then flatten.
        stock_plot_data = stock_data_full.T.flatten()  # Transpose (N, 1) -> (1, N) then flatten
        inflow_plot_data = inflow_data_full.T.flatten()  # Transpose (N, 1) -> (1, N) then flatten
        # Matrices are plotted as NxN, flatten directly.
        curve_plot_data = curve_data_full.flatten()
        cohort1_plot_data = cohort1_data_full.flatten()
        cohort2_plot_data = cohort2_data_full.flatten()

        # --- Update Heatmap Data using set_array ---
        # NaNs in the arrays will not be colored by the heatmap.
        collections["stock"].set_array(stock_plot_data)
        collections["inflow"].set_array(inflow_plot_data)
        collections["curve"].set_array(curve_plot_data)
        collections["cohort1"].set_array(cohort1_plot_data)
        collections["cohort2"].set_array(cohort2_plot_data)

        # --- Update dynamic titles ---
        stock_val = stock_data_full[time, 0]
        inflow_val = inflow_data_full[time, 0]
        axes["stock"].set_title(f"Stock = {stock_val}")  # Format value
        axes["inflow"].set_title(f"Inflow = {inflow_val}")  # Format value
        axes["cohort2"].set_title(f"Cohort Survival Matrix\n(Time step = {time})")

        # --- Update Equations using set_text (FAST) ---
        time_text_obj.set_text(f"Time step = {time}")
        eq_text_obj.set_text(equations[time])

        # Optional: fig.canvas.draw_idle() - uncomment if updates are jerky
        fig.canvas.draw_idle()

    # --- Create the slider ---
    step_slider = widgets.IntSlider(
        min=0, max=time_max - 1, value=0, description="Time step", continuous_update=False, layout={"width": "85%"}
    )

    # Connect the slider to the update function
    widgets.interactive_output(update_heatmap, {"time": step_slider})

    # --- Display ---
    display(step_slider)
    fig.tight_layout(pad=1.5)

    # Apply canvas settings from your code
    # Note: These might depend on the specific backend/version (like ipympl)
    # and might potentially conflict with rcParams settings if you used them.
    try:
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.toolbar_position = "right"
    except AttributeError:
        print("Warning: Canvas settings might not be available for this backend.")

    return fig


# --- End Core Implementations ---

# ---------------------------------------------------------------------------
# 4. Abstract Base Class (Strategy Interface)
# ---------------------------------------------------------------------------


class AnimationStrategy(abc.ABC):
    """Abstract Base Class for an animation strategy."""

    DEFAULT_FIGSIZE: Tuple[float, float] = (10, 10)  # Default fallback if needed

    @abc.abstractmethod
    def prepare_data(
        self,
        stock: Optional[NDArray],  # required for stock-driven strategy
        inflow: NDArray,
        curve_surv_matrix: NDArray,
        cohort_surv_matrix: NDArray,
        timesteps: range,
        **kwargs: Any,
    ) -> AnimationData:
        """Prepares the data structure required for this strategy's animation."""
        pass

    @abc.abstractmethod
    def create_animation(
        self, animation_data: AnimationData, figsize: Tuple[float, float], **kwargs: Any
    ) -> plt.Figure:
        """Creates the matplotlib Figure containing the animation setup."""
        pass


# ---------------------------------------------------------------------------
# 5. Concrete Strategy Classes
# ---------------------------------------------------------------------------


class FlowDrivenStrategy(AnimationStrategy):
    """Concrete strategy for flow-driven model animations."""

    # Define strategy-specific default
    DEFAULT_FIGSIZE: Tuple[float, float] = (10, 6)

    def prepare_data(
        self,
        stock: Optional[NDArray],  # Signature must match the ABC exactly
        inflow: NDArray,
        curve_surv_matrix: NDArray,
        cohort_surv_matrix: NDArray,
        timesteps: range,
        **kwargs: Any,
    ) -> AnimationData:
        # Delegates call to the specific implementation function
        # Does NOT pass the 'stock' parameter, as get_flow_driven... doesn't need it
        return get_flow_driven_animation_data(
            # stock=stock, # <--- Does NOT pass stock
            inflow=inflow,
            curve_surv_matrix=curve_surv_matrix,
            cohort_surv_matrix=cohort_surv_matrix,
            timesteps=timesteps,
            **kwargs,
        )

    def create_animation(
        self, animation_data: AnimationData, figsize: Tuple[float, float], **kwargs: Any
    ) -> plt.Figure:
        return flow_driven_animation(
            animation_data=animation_data,
            figsize=figsize,
            # **kwargs,  # DO NOT pass **kwargs here
        )


class StockDrivenStrategy(AnimationStrategy):
    """Concrete strategy for stock-driven model animations."""

    # Define strategy-specific default
    DEFAULT_FIGSIZE: Tuple[float, float] = (8, 10)

    def prepare_data(
        self,
        stock: Optional[NDArray],  # Signature must match the ABC exactly
        inflow: NDArray,
        curve_surv_matrix: NDArray,
        cohort_surv_matrix: NDArray,
        timesteps: range,
        **kwargs: Any,
    ) -> AnimationData:
        # Add check for stock here
        if stock is None:
            raise ValueError("Stock array must be provided for the stock-driven strategy.")

        # Delegates call to the specific implementation function, passing stock
        return get_stock_driven_animation_data(
            stock=stock,  # <--- Passes stock
            inflow=inflow,
            curve_surv_matrix=curve_surv_matrix,
            cohort_surv_matrix=cohort_surv_matrix,
            timesteps=timesteps,
            **kwargs,
        )

    def create_animation(
        self, animation_data: AnimationData, figsize: Tuple[float, float], **kwargs: Any
    ) -> plt.Figure:
        return stock_driven_animation(
            animation_data=animation_data,
            figsize=figsize,
            # **kwargs,  # DO NOT pass **kwargs here  # DO NOT pass **kwargs here
        )


# ---------------------------------------------------------------------------
# 6. Registry and Factory
# ---------------------------------------------------------------------------

# Registering the strategies in a dictionary for easy access
STRATEGY_REGISTRY: Dict[str, Type[AnimationStrategy]] = {
    "stock-driven": StockDrivenStrategy,
    "flow-driven": FlowDrivenStrategy,
}


# Factory function to retrieve the appropriate strategy based on model name
def get_strategy(model_name: str) -> AnimationStrategy:
    """
    Factory function to retrieve and instantiate an animation strategy
    based on the provided model name.
    """
    strategy_class = STRATEGY_REGISTRY.get(model_name)
    if not strategy_class:
        raise ValueError(
            f"Unknown model type: '{model_name}'. " f"Supported types are: {list(STRATEGY_REGISTRY.keys())}"
        )
    # Return an instance of the selected strategy class
    return strategy_class()


# ---------------------------------------------------------------------------
# 7. Main Orchestration Function
# ---------------------------------------------------------------------------


def interactive_animation(
    model_name: str,
    inflow: NDArray,
    curve_surv_matrix: NDArray,
    cohort_surv_matrix: NDArray,
    timesteps: range,
    figsize: Optional[Tuple[float, float]] = None,
    stock: Optional[NDArray] = None,  # Made Optional with default None
    **kwargs: Any,
):
    """
    Orchestrates the creation of a model-specific animation.

    1. Selects the appropriate strategy based on model_name using the factory.
    2. Uses the strategy object to prepare the animation data.
    3. Uses the strategy object to create the animation figure.

    Args:
        model_name: Name of the model (key in STRATEGY_REGISTRY).
        stock: Stock data array.
        inflow: Inflow data array.
        curve_surv_matrix: Survival curve matrix.
        cohort_surv_matrix: Cohort survival matrix.
        timesteps: Range object for timesteps.
        figsize: Figure size tuple for the animation plot.
        **kwargs: Additional keyword arguments passed down to the strategy's
            prepare_data and create_animation methods.

    Returns:
        The matplotlib Figure object containing the animation setup.
        Note: Interactive elements (widgets) created within the specific
            animation functions might need separate display handling
            by the caller, depending on the execution environment.
    """
    # 1. Select and instantiate the strategy
    strategy = get_strategy(model_name)

    # 2. Resolve figsize (before calling create_animation)
    # Use user-provided figsize if available, otherwise use the strategy's default
    actual_figsize: Tuple[float, float] = figsize if figsize is not None else strategy.DEFAULT_FIGSIZE

    # 2. Prepare data using the selected strategy
    print(f"Preparing data using {model_name} strategy.")
    animation_data = strategy.prepare_data(
        stock=stock,  # Pass whatever stock was given (could be None)
        inflow=inflow,
        curve_surv_matrix=curve_surv_matrix,
        cohort_surv_matrix=cohort_surv_matrix,
        timesteps=timesteps,
        **kwargs,
    )

    # 3. Create animation using the selected strategy
    print(f"Creating animation using {model_name} strategy.")
    # print(f"Display animation for {model_name}.")
    strategy.create_animation(animation_data=animation_data, figsize=actual_figsize, **kwargs)
