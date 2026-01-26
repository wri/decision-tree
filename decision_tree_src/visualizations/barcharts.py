import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib import patches as mpatches
from decision_tree_src.visualizations.visuals_config import DECISION_ORDER, DECISION_COLORS


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

def portfolio_breakdown_polygon(
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


def portfolio_breakdown_project(
    df: pd.DataFrame,
    baseline_col: str      = "baseline_project_label",
    ev_col: str            = "ev_project_label",
    figsize: tuple         = (14, 8),
    bar_height: float      = 0.6,
    annotate_offset: float = 0.01,
    fontsize: int          = 14,
    title: str             = None,
    style_kwargs: dict     = None
):
    """
    Grouped horizontal bar chart of PROJECT counts + proportions for each decision category.
    Bottom bar = ev_project_label
    Top bar    = baseline_project_label
    Colors use DECISION_COLORS[cat].

    Returns:
      fig, ax, counts, props
      where counts = {'baseline': counts_baseline, 'ev': counts_ev}
            props  = {'baseline': props_baseline, 'ev': props_ev}
    """
    # 1) counts & proportions for baseline and ev (ordered by DECISION_ORDER)
    counts_baseline = df[baseline_col].value_counts().reindex(DECISION_ORDER, fill_value=0)
    counts_ev       = df[ev_col].value_counts().reindex(DECISION_ORDER, fill_value=0)

    total_baseline = counts_baseline.sum()
    total_ev       = counts_ev.sum()

    props_baseline = counts_baseline / total_baseline if total_baseline > 0 else counts_baseline * 0
    props_ev       = counts_ev / total_ev if total_ev > 0 else counts_ev * 0

    cats = list(DECISION_ORDER)
    bar_colors = [DECISION_COLORS[cat] for cat in cats]

    # 2) positions for grouped bars (bottom = EV, top = Baseline)
    y = np.arange(len(cats))
    single_h = bar_height / 2.0
    y_ev       = y - single_h/2   # lower (bottom) bar
    y_baseline = y + single_h/2   # upper (top) bar

    # 3) plot grouped horizontal bars
    fig, ax = plt.subplots(figsize=figsize)
    bars_ev = ax.barh(
        y=y_ev,
        width=counts_ev.values,
        height=single_h,
        color=bar_colors,
        alpha=0.75,
        label="EV"
    )
    bars_baseline = ax.barh(
        y=y_baseline,
        width=counts_baseline.values,
        height=single_h,
        color=bar_colors,
        label="Baseline"
    )

    # 4) annotate percents for both bars
    total_max = max(total_baseline, total_ev)
    for bar, prop in zip(bars_ev, props_ev.values):
        ax.text(
            bar.get_width() + total_max * annotate_offset,
            bar.get_y() + bar.get_height()/2,
            f"{prop:.1%}",
            va="center",
            ha="left",
            fontsize=fontsize
        )
    for bar, prop in zip(bars_baseline, props_baseline.values):
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

    # 6) axis labels + styling (reuse your style_axis helper and defaults)
    default_style = dict(
        xlabel="Project Count",
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
        cnt = (df.groupby([project_col, col])[poly_col]
              .count()
              .reset_index(name="count")
        )
        tot = (df.groupby(project_col)[poly_col]
              .count()
              .reset_index(name="total")
        )
        merged = cnt.merge(tot, on=project_col)
        merged["prop"] = merged["count"] / merged["total"]
        p = (merged
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

def baseline_counts(
    df: pd.DataFrame,
    baseline_col: str        = "baseline_project_label",
    figsize: tuple           = (8, 6),        # slightly more square
    bar_height: float        = 0.6,
    fontsize: int            = 14,
    title: str               = " ",
    style_kwargs: dict       = None,
    annotate_offset: float   = 0.01,          # fraction of max count, used to offset labels
):
    """
    Horizontal bar chart of BASELINE project counts for three grouped categories:

        • remote  = {strong remote, weak remote}
        • field   = {strong field,  weak field}
        • unknown = {mangrove, review required, not available}

    X-axis: Project Count
    Y-axis: ['remote', 'field', 'unknown']

    Annotations show BOTH the proportion of the portfolio and the project count
    for each category, using the same color palette and axis styling helpers
    as your other charts (DECISION_COLORS, style_axis).

    Returns
    -------
    fig, ax, counts, props : (Figure, Axes, pd.Series, pd.Series)
        counts : Series indexed by ['remote','field','unknown'] with project counts.
        props  : Series with proportions in the same order.
    """
    # --- normalize & map baseline labels into grouped categories ---
    remote_set  = {"strong remote", "weak remote"}
    field_set   = {"strong field",  "weak field"}
    unknown_set = {"mangrove", "review required", "not available"}

    labels = (
        df[baseline_col]
        .dropna()
        .map(lambda x: str(x).strip().lower())
    )

    def to_group(lbl: str) -> str:
        if lbl in remote_set:
            return "remote"
        if lbl in field_set:
            return "field"
        if lbl in unknown_set:
            return "unknown"
        # any unexpected value falls back to 'unknown'
        return "unknown"

    grouped = labels.map(to_group)

    # --- counts and proportions in desired order ---
    order = ["remote", "field", "unknown"]
    counts = grouped.value_counts().reindex(order, fill_value=0)
    total  = counts.sum()
    props  = counts / total if total > 0 else counts * 0.0

    # --- colors from DECISION_COLORS (fallbacks included) ---
    remote_color  = (
        DECISION_COLORS.get("strong remote")
        or DECISION_COLORS.get("weak remote")
        or "#6baed6"
    )
    field_color   = (
        DECISION_COLORS.get("strong field")
        or DECISION_COLORS.get("weak field")
        or "#fd8d3c"
    )
    unknown_color = (
        DECISION_COLORS.get("not available")
        or DECISION_COLORS.get("mangrove")
        or DECISION_COLORS.get("review required")
        or "#9e9e9e"
    )
    colors = [remote_color, field_color, unknown_color]

    # --- plot horizontal bars ---
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(y=y, width=counts.values, height=bar_height, color=colors)

    # y-axis labels & order (remote at top)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()

    # --- annotate: "<count> (pp%)" to the right of each bar ---
    max_count = max(counts.max(), 1)
    x_offset = max_count * annotate_offset
    for bar, cnt, prop in zip(bars, counts.values, props.values):
        ax.text(
            x=bar.get_width() + x_offset,
            y=bar.get_y() + bar.get_height() / 2,
            s=f"{cnt} ({prop:.0%})",
            va="center",
            ha="left",
            fontsize=fontsize
        )

    # Optionally expand xlim a bit so annotations fit
    ax.set_xlim(0, counts.max() * (1 + 3 * annotate_offset) if counts.max() > 0 else 1)

    # --- axis labels + styling (reuse your helper) ---
    default_style = dict(
        xlabel="Project Count",
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
    return fig, ax, counts, props


def plot_project_scores_dumbbell(
    df: pd.DataFrame,
    project_names: list,
    baseline_col: str = "baseline_project_score_0_100",
    ev_col: str       = "ev_project_score_0_100",
    label_col: str    = "project_name",
    figsize: tuple    = None,
    title: str        = " ",
    style_kwargs: dict = None,
    annotate: bool     = True,       # annotate with "baseline"/"ev" strings
    label_offset: float = 0.5        # x-offset (data units) for annotation text
):
    """
    Horizontal plot where EACH project row shows a faint line from 0–100
    and two markers placed at the baseline and EV scores.

    Color coding is based on the score using the following buckets:
        >= 70  => "strong remote"
        55–70  => "weak remote"
        30–55  => "weak field"
        <  30  => "strong field"

    Annotations (if enabled) label each point with "baseline" or "ev"
    (category name, not the numeric score).

    Returns
    -------
    None
    """

    # --- helper: score -> decision bucket (for color lookup) ---
    def score_to_bucket(x: float) -> str:
        if pd.isna(x):
            return "strong field"  # fallback; will be skipped anyway
        if x >= 70:
            return "strong remote"
        if x >= 55:
            return "weak remote"
        if x >= 30:
            return "weak field"
        return "strong field"

    # --- subset & order by the provided project list (preserve order) ---
    prj_set = set(project_names)
    used_df = df[df[label_col].isin(prj_set)].copy()

    # Keep only requested projects and in the exact order provided
    used_df[label_col] = used_df[label_col].astype(str)
    used_df = used_df.set_index(label_col).reindex(project_names).reset_index()

    # Coerce to numeric
    used_df[baseline_col] = pd.to_numeric(used_df[baseline_col], errors="coerce")
    used_df[ev_col]       = pd.to_numeric(used_df[ev_col], errors="coerce")

    # --- notify about rows that can't be plotted BEFORE dropping them ---
    missing_mask = used_df[baseline_col].isna() | used_df[ev_col].isna()
    if missing_mask.any():
        for _, row in used_df.loc[missing_mask, [label_col, baseline_col, ev_col]].iterrows():
            print(f"[skip] {row[label_col]} does not contain enough information to be plotted "
                  f"(baseline={row[baseline_col]}, ev={row[ev_col]}).")

    # Drop rows missing either score
    used_df = used_df.dropna(subset=[baseline_col, ev_col])

    n = len(used_df)
    if n == 0:
        raise ValueError("No valid projects to plot after filtering and numeric coercion.")

    # --- figure sizing: auto height if not provided ---
    if figsize is None:
        figsize = (12, max(3.0, 0.55 * n + 1.5))

    def color_for_score(x: float) -> str:
        bucket = score_to_bucket(x)
        return DECISION_COLORS.get(bucket)

    refline_color  = "#bdbdbd"  # neutral reference line per row

    # --- y positions (first project at top) ---
    y = np.arange(n)[::-1]                 # invert so first item is at the top
    ylabels = used_df[label_col].tolist()[::-1]
    x_base  = used_df[baseline_col].to_numpy()[::-1]
    x_ev    = used_df[ev_col].to_numpy()[::-1]

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # full 0–100 reference line per project row (faint)
    for yi in y:
        ax.hlines(y=yi, xmin=0, xmax=100, linewidth=1.5, color=refline_color, alpha=0.4, zorder=1)

    # scatter markers, colored by score buckets
    base_colors = [color_for_score(v) for v in x_base]
    ev_colors   = [color_for_score(v) for v in x_ev]

    ax.scatter(x_base, y, s=40, color=base_colors, zorder=3)
    ax.scatter(x_ev,   y, s=40, color=ev_colors,   zorder=3)

    # optional annotations: label markers with "baseline" / "ev"
    if annotate:
        for xi, yi in zip(x_base, y):
            ax.text(xi + label_offset, yi, "baseline", va="center", ha="left", fontsize=11)
        for xi, yi in zip(x_ev, y):
            ax.text(xi + label_offset, yi, "ev", va="center", ha="left", fontsize=11)

    # axes
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # match your house style
    default_style = dict(
        xlabel="Score",
        ylabel=" ",
        title=title,
        x_grid=True,
        y_grid=False,
        hide_bottom=False,
        fontsize=12
    )
    style_args = {**default_style, **(style_kwargs or {})}
    style_axis(ax, **style_args)

    plt.tight_layout()
    return None
