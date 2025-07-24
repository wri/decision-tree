import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from visuals_config import DECISION_COLORS, DECISION_ORDER


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


def plot_decision_proportions(
    df: pd.DataFrame,
    project_col: str      = "project_name",
    poly_col: str         = "poly_id",
    decision_col: str     = "decision",
    sort_by: str          = None,
    bar_width: float      = 0.6,
    figsize: tuple        = (20, 10),
    style_kwargs: dict    = None
):
    """
    Compute & plot, for each project, the proportion of polygons assigned to each decision,
    then apply your style_axis() styling.

    style_kwargs is passed directly to style_axis(ax, **style_kwargs).
    """
    # 1) count per project & decision
    cnt = (
        df.groupby([project_col, decision_col])[poly_col]
          .count()
          .reset_index(name="count")
    )

    # 2) total per project
    tot = (
        df.groupby(project_col)[poly_col]
          .count()
          .reset_index(name="total")
    )

    # 3) merge & compute proportions
    merged = pd.merge(cnt, tot, on=project_col)
    merged["proportion"] = merged["count"] / merged["total"]

    # 4) pivot to wide format
    pivot = (
        merged
        .pivot(index=project_col, columns=decision_col, values="proportion")
        .fillna(0)
    )

    # 5) optional sorting
    if sort_by and sort_by in pivot.columns:
        pivot = pivot.sort_values(by=sort_by, ascending=False)

    colors = [DECISION_COLORS[dec] for dec in pivot.columns]

    # 7) plot
    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(
        kind="bar",
        stacked=True,
        color=colors,
        width=bar_width,
        ax=ax,
        edgecolor="none"
    )

    # 8) legend just outside
    ax.legend(
        title="Decision",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=12
    )

    # 9) style the axes -- you can override any of these by passing style_kwargs
    default_style = dict(
        xlabel="Project",
        ylabel="Proportion",
        title="Decision Proportions by Project",
        y_grid=True,
        x_grid=False,
        hide_bottom=False,
        fontsize=12      # smaller ticks to fit 81 labels
    )
    style_args = {**default_style, **(style_kwargs or {})}
    style_axis(ax, **style_args)

    plt.tight_layout()
    #return pivot, ax


def plot_decision_hbar(
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