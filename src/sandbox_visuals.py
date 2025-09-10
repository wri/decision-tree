
"""
verification_viz_functions.py
---------------------------------
Function-based plotting library for portfolio verification insights.
"""

from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re

# ------------------------------
# Consistent axis styling helper (from user)
# ------------------------------
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
    """Applies consistent styling to the axes, including labels and gridlines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if hide_bottom:
        ax.spines['bottom'].set_visible(False)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    if y_grid:
        ax.yaxis.grid(True, color=grid_color); ax.set_axisbelow(True)
    if x_grid:
        ax.xaxis.grid(True, color=grid_color); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, labelpad=15, fontsize=fontsize)
    if ylabel: ax.set_ylabel(ylabel, labelpad=15, fontsize=fontsize)
    if tick_format: ax.ticklabel_format(style='plain', axis='y')
    if title:
        title_fontsize = fontsize + 2 if fontsize else 14
        ax.set_title(title, fontsize=title_fontsize, pad=15)
    if fontsize:
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

# ------------------------------
# Internal helpers
# ------------------------------
def _classify(series: pd.Series) -> pd.Series:
    s = series.fillna('')
    return pd.Series(
        np.where(s.str.contains('field', case=False), 'field',
        np.where(s.str.contains('remote', case=False), 'remote', 'other')),
        index=series.index
    )

def _ensure_categories(df: pd.DataFrame) -> pd.DataFrame:
    '''Creates general remote/field/other category for each poly based on string search'''
    d = df.copy()
    if 'baseline_category' not in d.columns:
        d['baseline_category'] = _classify(d['baseline_decision'])
    if 'ev_category' not in d.columns:
        d['ev_category'] = _classify(d['ev_decision'])
    return d

def _past_subset(df: pd.DataFrame,
                 cutoff: Union[str, pd.Timestamp] = "2025-12",
                 plantstart_col: str = "plantstart",
                 dayfirst: bool = False) -> pd.DataFrame:
    """Return rows where both Baseline and EV months are <= the cutoff month.

    Baseline = plantstart - 1 year; EV = plantstart + 2 years.
    Month-level comparison uses to_period('M'), which respects month & year.

    Parameters
    ----------
    cutoff : str | pd.Timestamp, default "2025-12"
        Accepts "YYYY-MM", "YYYY-MM-DD", or Timestamp. Compared at month granularity.
    plantstart_col : str
        Name of the plantstart column (string or datetime-like).
    dayfirst : bool
        Use True if your dates are DD/MM/YY.

    Returns
    -------
    pd.DataFrame
    """
    d = df.copy()
    if not np.issubdtype(d[plantstart_col].dtype, np.datetime64):
        d[plantstart_col] = pd.to_datetime(d[plantstart_col], 
                                           dayfirst=dayfirst, 
                                           errors='coerce')
        
    baseline_date = d[plantstart_col] - pd.DateOffset(years=1)
    ev_date = d[plantstart_col] + pd.DateOffset(years=2)

    if isinstance(cutoff, str):
        cutoff_ts = pd.to_datetime(cutoff + '-01' if re.fullmatch(r'\d{4}-\d{2}', cutoff) else cutoff, errors='coerce')
    else:
        cutoff_ts = pd.to_datetime(cutoff)
    cutoff_period = cutoff_ts.to_period('M')

    mask = (baseline_date.dt.to_period('M') <= cutoff_period) & (ev_date.dt.to_period('M') <= cutoff_period)
    return d.loc[mask].copy()

# ------------------------------
# 1) Portfolio 100% bars: Field vs Remote
# ------------------------------
def plot_portfolio_share(df: pd.DataFrame,
                         figsize: Tuple[float,float] = (7,5),
                         title: Optional[str] = None,
                         fontsize: Optional[int] = 12):
    """Stacked 100% bars for Field vs Remote at Baseline and EV."""
    d = _ensure_categories(df)

    def share(col: str):
        v = d[d[col].isin(['field','remote'])][col].value_counts().reindex(['field','remote']).fillna(0)
        tot = v.sum()
        return (v['field']/tot if tot>0 else np.nan, v['remote']/tot if tot>0 else np.nan)

    f_b, r_b = share('baseline_category')
    f_e, r_e = share('ev_category')

    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(2)
    field_props = [f_b, f_e]
    remote_props = [r_b, r_e]
    ax.bar(xs, field_props, label='Field')
    ax.bar(xs, remote_props, bottom=field_props, label='Remote')
    ax.set_xticks(xs); ax.set_xticklabels(['Baseline','EV'])
    ax.set_ylim(0,1)
    for i,(f,r) in enumerate(zip(field_props, remote_props)):
        if pd.notnull(f):
            ax.text(i, f/2, f"{f*100:.0f}%", ha="center", va="center", fontsize=(fontsize or 12))
            ax.text(i, f + r/2, f"{r*100:.0f}%", ha="center", va="center", fontsize=(fontsize or 12))
    ax.axhline(0.8, linestyle='--')
    style_axis(ax, ylabel='Share of polygons', title=title or 'Portfolio Share of Decisions: Field vs Remote', fontsize=fontsize)
    ax.legend()
    fig.tight_layout()
    return fig, ax

# ------------------------------
# 2) EV leaderboard (past only)
# ------------------------------
def plot_ev_field_share_by_project_past(df: pd.DataFrame,
                                        figsize: Tuple[float,float] = None,
                                        title: Optional[str] = None,
                                        fontsize: Optional[int] = 12,
                                        year_cutoff: int = 2025):
    """Horizontal bar chart of EV % Field per project for rows with Baseline & EV months <= Dec year_cutoff."""
    d = _ensure_categories(df)
    past = _past_subset(d, cutoff=f"{year_cutoff}-12")
    v = (past[past['ev_category'].isin(['field','remote'])]
            .groupby(['project_id','project_name'])['ev_category']
            .value_counts()
            .unstack(fill_value=0))
    v['total'] = v.sum(axis=1)
    v = v[v['total']>0]
    v['pct_field'] = v['field']/v['total']
    v = v.sort_values('pct_field', ascending=False).reset_index()

    if figsize is None:
        figsize = (9, max(4, 0.25*len(v)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(v['project_name'], v['pct_field'])
    ax.invert_yaxis(); ax.set_xlim(0,1)
    for i, val in enumerate(v['pct_field']):
        ax.text(val + 0.01, i, f"{val*100:.0f}%", va='center', fontsize=(fontsize or 12))
    ax.axvline(0.8, linestyle='--')
    style_axis(ax, xlabel='Share of polygons assigned to Field at EV',
               title=title or f'Projects with EV already passed (≤{year_cutoff})',
               x_grid=True, fontsize=fontsize)
    fig.tight_layout()
    return fig, ax

# ------------------------------
# 3) Baseline leaderboard (past only)
# ------------------------------
def plot_baseline_field_share_by_project_past(df: pd.DataFrame,
                                              figsize: Tuple[float,float] = None,
                                              title: Optional[str] = None,
                                              fontsize: Optional[int] = 12,
                                              year_cutoff: int = 2025):
    """Horizontal bar chart of Baseline % Field per project for rows with Baseline & EV months <= Dec year_cutoff."""
    d = _ensure_categories(df)
    past = _past_subset(d, cutoff=f"{year_cutoff}-12")
    v = (past[past['baseline_category'].isin(['field','remote'])]
            .groupby(['project_id','project_name'])['baseline_category']
            .value_counts()
            .unstack(fill_value=0))
    v['total'] = v.sum(axis=1)
    v = v[v['total']>0]
    v['pct_field'] = v['field']/v['total']
    v = v.sort_values('pct_field', ascending=False).reset_index()

    if figsize is None:
        figsize = (9, max(4, 0.25*len(v)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(v['project_name'], v['pct_field'])
    ax.invert_yaxis(); ax.set_xlim(0,1)
    for i, val in enumerate(v['pct_field']):
        ax.text(val + 0.01, i, f"{val*100:.0f}%", va='center', fontsize=(fontsize or 12))
    ax.axvline(0.8, linestyle='--')
    style_axis(ax, xlabel='Share of polygons assigned to Field at Baseline',
               title=title or f'Projects with Baseline already passed (≤{year_cutoff})',
               x_grid=True, fontsize=fontsize)
    fig.tight_layout()
    return fig, ax

# ------------------------------
# 4) Risk map (past only)
# ------------------------------
def plot_risk_map(df: pd.DataFrame,
                  figsize: Tuple[float,float] = (7,6),
                  title: Optional[str] = None,
                  fontsize: Optional[int] = 12,
                  year_cutoff: int = 2025,
                  label_top_n: int = 10):
    """Scatter of Baseline vs EV % Field per project; bubble size = # EV polys."""
    d = _ensure_categories(df)
    # remove past subset bc polygons shouldnt be dropped
    # this could still be plotted 
    past = _past_subset(d, cutoff=f"{year_cutoff}-12")

    ev = (past[past['ev_category'].isin(['field','remote'])]
            .groupby(['project_id','project_name'])['ev_category']
            .value_counts()
            .unstack(fill_value=0))
    ev['ev_total'] = ev.sum(axis=1)
    ev['ev_pct_field'] = ev['field']/ev['ev_total']

    bl = (past[past['baseline_category'].isin(['field','remote'])]
            .groupby(['project_id','project_name'])['baseline_category']
            .value_counts()
            .unstack(fill_value=0))
    bl['bl_total'] = bl.sum(axis=1)
    bl['bl_pct_field'] = bl['field']/bl['bl_total']

    m = (pd.concat([ev[['ev_total','ev_pct_field']],
                    bl[['bl_total','bl_pct_field']]], axis=1)
            .fillna(0)
            .reset_index())

    fig, ax = plt.subplots(figsize=figsize)

    # # --- configurable sizing ---
    # size_col   = 'ev_total'   # or 'bl_total'
    # size_scale = 7            # multiply the raw size by this factor
    # size_min   = 20           # min area in points^2
    # size_max   = 600          # max area in points^2
    # use_sqrt   = True         # area-perception friendly scaling

    # raw = m[size_col].fillna(0).to_numpy()
    # if use_sqrt:
    #     # preserves relative *area* perception better than linear
    #     sizes = np.clip(np.sqrt(raw) * size_scale, size_min, size_max)
    # else:
    #     sizes = np.clip(raw * size_scale, size_min, size_max)

    # --- scatter with circles ---
    ax.scatter(
        m['bl_pct_field'], m['ev_pct_field'],
        #s=sizes,
        marker='o',          
        alpha=0.7,
        linewidths=0,
        edgecolors='none'
    )
    ax.plot([0,1],[0,1]) 
    ax.axhline(0.8, linestyle='--')
    ax.axvline(0.8, linestyle='--')
    ax.set_xlim(0,1.1)
    ax.set_ylim(0,1.1)

    # Label all points with EV % field ≥ 0.8 OR Baseline % field ≥ 0.8
    label_df = m[(m['ev_pct_field'] >= 0.8) | (m['bl_pct_field'] >= 0.8)]

    for _, row in label_df.iterrows():
        ax.text(
            row['bl_pct_field'],
            row['ev_pct_field'],
            row['project_name'],
            fontsize=(fontsize or 10),
            zorder=4,
            clip_on=False  
        )
        
    style_axis(ax, 
               xlabel='Baseline: % Field', 
               ylabel='EV: % Field',
               title=title or f'Project Risk Map (years ≤ {year_cutoff})',
               x_grid=True, 
               y_grid=True, 
               fontsize=fontsize)
    
    fig.tight_layout()
    return fig, ax

# ------------------------------
# 5) Field demand by year
# ------------------------------
def plot_field_demand_by_year(df: pd.DataFrame,
                              figsize: Tuple[float,float] = (8,5),
                              title: Optional[str] = None,
                              fontsize: Optional[int] = 12):
    """Side-by-side bars of # polygons requiring Field by year for Baseline vs EV."""
    d = _ensure_categories(df)
    f_by_bl = d[d['baseline_category']=='field'].groupby('baseline_year').size()
    f_by_ev = d[d['ev_category']=='field'].groupby('ev_year').size()
    years = sorted(set(f_by_bl.index).union(set(f_by_ev.index)))
    x = np.arange(len(years)); bar_w = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - bar_w/2, [f_by_bl.get(y,0) for y in years], width=bar_w, label='Baseline')
    ax.bar(x + bar_w/2, [f_by_ev.get(y,0) for y in years], width=bar_w, label='EV')
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.legend()
    style_axis(ax, ylabel='# polygons requiring Field',
               title=title or 'Field Crew Demand Forecast by Year',
               y_grid=True, fontsize=fontsize)
    fig.tight_layout()
    return fig, ax
