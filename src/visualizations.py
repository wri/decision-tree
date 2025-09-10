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
    fontsize: int           = 14,
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
        fontsize=fontsize
    )
    # merge user overrides
    style_args = {**default_style, **(style_kwargs or {})}
    style_axis(ax, **style_args)
    plt.tight_layout()

def NEW_portfolio_breakdown(
    df: pd.DataFrame,
    baseline_col: str     = "baseline_decision",
    ev_col: str           = "ev_decision",
    figsize: tuple        = (14, 8),
    bar_height: float     = 0.6,
    annotate_offset: float= 0.01,
    fontsize: int         = 14,
    title: str            = None,
    style_kwargs: dict    = None
):
    """
    Grouped horizontal bar chart of counts + proportions for each decision category.
    Bottom bar = baseline_decision
    top bar = ev_decision.
    Colors use DECISION_COLORS[cat].

    Returns:
      fig, ax, counts, props
      where counts = {'baseline': counts_baseline, 'ev': counts_ev}
            props  = {'baseline': props_baseline, 'ev': props_ev}
    """
    # 1) counts & proportions for baseline and ev
    counts_baseline = df[baseline_col].value_counts().reindex(DECISION_ORDER, fill_value=0)
    counts_ev       = df[ev_col].value_counts().reindex(DECISION_ORDER, fill_value=0)

    total_baseline = counts_baseline.sum()
    total_ev       = counts_ev.sum()

    props_baseline = counts_baseline / total_baseline if total_baseline > 0 else counts_baseline*0
    props_ev       = counts_ev / total_ev if total_ev > 0 else counts_ev*0

    cats = list(DECISION_ORDER)
    bar_colors = [DECISION_COLORS[cat] for cat in cats]

    # 2) positions for grouped bars (bottom = baseline, top = ev)
    y = np.arange(len(cats))
    single_h = bar_height / 2.0   # split the group height across two bars
    y_baseline = y - single_h/2   # lower (bottom) bar
    y_ev       = y + single_h/2   # upper (top) bar

    # 3) plot grouped horizontal bars
    fig, ax = plt.subplots(figsize=figsize)
    bars_baseline = ax.barh(
        y=y_baseline,
        width=counts_baseline.values,
        height=single_h,
        color=bar_colors,
        label="Baseline"
    )
    bars_ev = ax.barh(
        y=y_ev,
        width=counts_ev.values,
        height=single_h,
        color=bar_colors,
        alpha=0.75,         # slight distinction, same palette
        label="EV"
    )

    # 4) annotate counts & percents for both bars
    total_max = max(total_baseline, total_ev)
    for bar, cnt, prop in zip(bars_baseline, counts_baseline.values, props_baseline.values):
        ax.text(
            bar.get_width() + total_max * annotate_offset,
            bar.get_y() + bar.get_height()/2,
            f"{prop:.1%}",
            va="center",
            ha="left",
            fontsize=fontsize
        )
    for bar, cnt, prop in zip(bars_ev, counts_ev.values, props_ev.values):
        ax.text(
            bar.get_width() + total_max * annotate_offset,
            bar.get_y() + bar.get_height()/2,
            f"{prop:.1%}",
            va="center",
            ha="left",
            fontsize=fontsize
        )

    # 5) category labels & order (first in DECISION_ORDER at the top)
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()

    # 6) style + legend
    default_style = dict(
        xlabel="Polygon Count",
        ylabel=" ",
        title=title,
        x_grid=True,
        y_grid=False,
        hide_bottom=False,
        fontsize=fontsize
    )
    style_args = {**default_style, **(style_kwargs or {})}
    style_axis(ax, **style_args)

    plt.tight_layout()
    counts = {"baseline": counts_baseline, "ev": counts_ev}
    props  = {"baseline": props_baseline, "ev": props_ev}
    return None

def plot_decision_proportions(df: pd.DataFrame,
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
                edgecolor="white",
                #linewidth=0.7
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
                edgecolor="white",
                #linewidth=0.7
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


def plot_decision_proportions_faceted(
    df: pd.DataFrame,
    project_col: str   = "project_name",
    poly_col: str      = "poly_id",
    ncols: int         = 3,
    projects: list     = None,
    figsize: tuple     = (24, 12),
    title: str         = None,
    annotate: bool     = False,
    label_min_prop: float = 0.25,
    style_kwargs: dict = None
):
    """
    Grid of small subplots; each project shows two stacked bars (Baseline, EV).
    Returns: (fig, axes, pivot_base, pivot_ev)
    """

    # --- pivot logic mirrors your function ---
    def make_pivot(source, col):
        cnt = (
            source.groupby([project_col, col])[poly_col]
                  .count()
                  .reset_index(name="count")
        )
        tot = (
            source.groupby(project_col)[poly_col]
                  .count()
                  .reset_index(name="total")
        )
        merged = cnt.merge(tot, on=project_col)
        merged["prop"] = merged["count"] / merged["total"]
        p = merged.pivot(index=project_col, columns=col, values="prop").fillna(0)
        counts = merged.pivot(index=project_col, columns=col, values="count").fillna(0)
        totals = tot.set_index(project_col)["total"]
        return p, counts, totals

    pivot_base, counts_base, totals = make_pivot(df, "baseline_decision")
    pivot_ev,   counts_ev,   _      = make_pivot(df, "ev_decision")

    cats = [c for c in DECISION_ORDER if c in pivot_base.columns or c in pivot_ev.columns]
    pivot_base = pivot_base.reindex(columns=cats, fill_value=0)
    pivot_ev   = pivot_ev.reindex(columns=cats, fill_value=0)
    counts_base = counts_base.reindex(columns=cats, fill_value=0)
    counts_ev   = counts_ev.reindex(columns=cats, fill_value=0)

    all_projects = pivot_base.index.tolist()
    if projects is None:
        projects = all_projects

    n = len(projects)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    ax_list = axes.ravel()

    for idx, proj in enumerate(projects):
        ax = ax_list[idx]
        # stacked vertical bars at x=0 (Baseline) and x=1 (EV)
        x_positions = [0, 1]
        bar_width = 0.6

        # Baseline stack
        bottom = 0
        for cat in cats:
            h = pivot_base.loc[proj, cat]
            ax.bar(x_positions[0], h, width=bar_width, bottom=bottom, color=DECISION_COLORS[cat], edgecolor="white")
            if annotate and h >= label_min_prop:
                cnt = int(counts_base.loc[proj, cat])
                ax.text(x_positions[0], bottom + h/2, f"{cnt} ({h:.0%})", ha="center", va="center", fontsize=9)
            bottom += h

        # EV stack
        bottom = 0
        for cat in cats:
            h = pivot_ev.loc[proj, cat]
            ax.bar(x_positions[1], h, width=bar_width, bottom=bottom, color=DECISION_COLORS[cat], edgecolor="white")
            if annotate and h >= label_min_prop:
                cnt = int(counts_ev.loc[proj, cat])
                ax.text(x_positions[1], bottom + h/2, f"{cnt} ({h:.0%})", ha="center", va="center", fontsize=9)
            bottom += h

        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Baseline", "EV"])
        ax.set_ylim(0, 1)
        style_axis(ax, xlabel="", ylabel="", title=proj, y_grid=True, x_grid=False, fontsize=10)

    # hide any empty panels
    for j in range(idx + 1, len(ax_list)):
        ax_list[j].axis("off")

    # figure-level legend
    handles = [mpatches.Patch(color=DECISION_COLORS[c], label=c) for c in cats]
    fig.legend(handles=handles, labels=cats, title="Decision", loc="upper right", bbox_to_anchor=(0.99, 0.98))

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout(rect=[0, 0, 0.86, 0.98])
    return None

