import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib import patches as mpatches
from visuals_config import DECISION_ORDER, DECISION_COLORS


def style_axis(ax, 
               xlabel: str = None, 
               ylabel: str = None, 
               title: str = None,
               y_grid: bool = None,
               x_grid: bool = None,
               hide_bottom: bool = None,
               grid_color: str = '#DDDDDD',
               tick_format: str = None,
               fontsize: int = None):
    """
    Applies consistent styling to the axes, including labels and gridlines.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to be styled.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    y_grid (bool): Option to add gridlines to the y-axis.
    x_grid (bool): Option to add gridlines to the x-axis.
    grid_color (str, optional): The color of the gridlines. Default is '#DDDDDD'.
    tick_format (str, optional): Optionally set tick label format for y-axis.
    fontsize (int, optional): Font size for axis labels and tick marks.
    """
    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    if hide_bottom:
        ax.spines['bottom'].set_visible(False)

    # Style the bottom and left spines
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    #ax.spines['right'].set_color(grid_color)
    #ax.spines['left'].set_color(grid_color)
    
    # Set gridlines and axis below the plot elements
    if y_grid:
        ax.yaxis.grid(True, color=grid_color)
        ax.set_axisbelow(True)
    if x_grid:
        ax.xaxis.grid(True, color=grid_color)
        ax.set_axisbelow(True)
    
    # Set the axis labels
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=15, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=15, fontsize=fontsize)
    if tick_format:
        ax.ticklabel_format(style='plain', axis='y')
    
    # Set title with a slightly larger font size
    if title:
        title_fontsize = fontsize + 2 if fontsize else 14
        ax.set_title(title, fontsize=title_fontsize, pad=15)
    
    # Set tick font sizes
    if fontsize:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

def portfolio_breakdown(
    df: pd.DataFrame,
    decision_col: str       = "decision",
    figsize: tuple          = (14, 8),
    bar_height: float       = 0.6,
    annotate_offset: float  = 0.01,
    fontsize: int           = 10,
    title: str = None,
    style_kwargs: dict      = None
):
    """
    Plot a horizontal bar chart of counts + proportions for each category in `decision_col`,
    then apply your style_axis() formatting.

    Returns:
      fig, ax, counts, props
    """
    counts = df[decision_col].value_counts()
    counts = counts.reindex(DECISION_ORDER, fill_value=0)
    total = counts.sum()
    props = counts / total

    cats = counts.index.tolist()
    bar_colors = [DECISION_COLORS[cat] for cat in counts.index]

    # 3) plot horizontal bars
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        y=cats,
        width=counts.values,
        height=bar_height,
        color=bar_colors
    )

    # 4) annotate count & percent
    for bar, cnt, prop in zip(bars, counts, props):
        ax.text(
            bar.get_width() + total * annotate_offset,
            bar.get_y() + bar.get_height()/2,
            f"{cnt} ({prop:.1%})",
            va="center",
            ha="left",
            fontsize=14
        )

    # 5) invert so first in `order` is at top
    ax.invert_yaxis()

    # 6) apply style_axis for consistent formatting
    #   default styling for horizontal bars:
    default_style = dict(
        xlabel="Polygon Count",
        ylabel=" ",
        title=title,
        x_grid=True,       # vertical gridlines
        y_grid=False,
        hide_bottom=False,
        fontsize=14
    )
    # merge user overrides
    style_args = {**default_style, **(style_kwargs or {})}
    style_axis(ax, **style_args)

    plt.tight_layout()

def plot_decision_proportions(
    df: pd.DataFrame,
    project_col: str     = "project_name",
    poly_col: str        = "poly_id",
    sort_by              = None,
    group_height: float  = 0.8,
    figsize: tuple       = (24, 12),
    title: str           = None,
    threshold: float     = None,   
    style_kwargs: dict   = None
):
    """
    For each project, draws two side-by-side stacked bars (baseline vs EV) 
    of decision proportions.
    Returns (pivot_base, pivot_ev, fig, ax).
    """
    def make_pivot(col):
        cnt = (
            df.groupby([project_col, col])[poly_col]
              .count()
              .reset_index(name="count")
        )
        tot = (
            df.groupby(project_col)[poly_col]
              .count()
              .reset_index(name="total")
        )
        merged = cnt.merge(tot, on=project_col)
        merged["prop"] = merged["count"] / merged["total"]
        p = (
            merged
            .pivot(index=project_col, columns=col, values="prop")
            .fillna(0)
        )
        return p

    # 1) build both pivots
    pivot_base = make_pivot("baseline_decision")
    pivot_ev   = make_pivot("ev_decision")

    # 2) unify columns in canonical order
    cats = [c for c in DECISION_ORDER if c in pivot_base.columns or c in pivot_ev.columns]
    pivot_base = pivot_base.reindex(columns=cats, fill_value=0)
    pivot_ev   = pivot_ev.reindex(columns=cats, fill_value=0)

    # 3) optional multi-key sort on baseline, then apply to EV
    if sort_by:
        # normalize to list
        keys = sort_by if isinstance(sort_by, (list, tuple)) else [sort_by]
        # keep only valid decision columns
        valid = [k for k in keys if k in pivot_base.columns]
        if valid:
            # sort baseline on all keys (descending)
            pivot_base = pivot_base.sort_values(
                by=valid,
                ascending=[False]*len(valid)
            )
            # reindex EV to match
            pivot_ev = pivot_ev.reindex(pivot_base.index)

    projects = pivot_base.index.tolist()
    n = len(projects)
    y = np.arange(n)

    # 4) bar geometry
    half = group_height / 2
    offset = half / 2

    # 5) plot
    fig, ax = plt.subplots(figsize=figsize)
    for i, proj in enumerate(projects):
        # baseline stacked bar
        left = 0
        for cat in cats:
            width = pivot_base.loc[proj, cat]
            ax.barh(
                y[i] + offset,
                width,
                height=half,
                left=left,
                color=DECISION_COLORS[cat],
                edgecolor="black",
                linewidth=0.7
            )
            left += width

        # ev stacked bar
        left = 0
        for cat in cats:
            width = pivot_ev.loc[proj, cat]
            ax.barh(
                y[i] - offset,
                width,
                height=half,
                left=left,
                color=DECISION_COLORS[cat],
                edgecolor="black",
                linewidth=0.7
            )
            left += width

    if threshold is not None:
        ax.axvline(
            x=threshold,
            color="red",
            linestyle="--",
            linewidth=2
        )

    handles = [mpatches.Patch(color=DECISION_COLORS[c], label=c) for c in cats]
    ax.legend(
        handles,
        cats,
        title="Decision",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=12
    )

    # 7) ticks & labels
    ax.set_yticks(y)
    ax.set_yticklabels(projects)
    ax.margins(y=0)
    if title:
        style_kwargs = style_kwargs or {}
        style_kwargs.setdefault("title", title)

    # 8) style axes
    default_style = dict(
        xlabel="Proportion",
        ylabel="",
        y_grid=True,
        x_grid=False,
        hide_bottom=False,
        fontsize=16
    )
    style_axis(ax, **{**default_style, **(style_kwargs or {})})

    plt.tight_layout()



