import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timedelta

def img_avail_hist(df, bin_width=10):
    """
    Groups projects into bins based on imagery coverage percent and prints a count per bin.
    Plots a histogram of the image availability distribution.

    Args:
        df (pd.DataFrame): Dataframe containing the image availability by project
        bin_width (int): Width of each bin (e.g., 10 for 0-10%, 10-20%, etc...)
    """
    # Define the bin edges
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101] # Right now slightly hack-y
    
    # Create labels for the bins
    labels = [f"{i}-{i+10}%" for i in range(0, 100, 10)]

    # Bin the data
    df['coverage_bin'] = pd.cut(df['total_percent_area_covered'], bins=bins, labels=labels, include_lowest=True, right=False) # Includes 0 in 0-10% cat. Includes 100, but not 90 in 90-100% cat)
    bin_counts = df['coverage_bin'].value_counts().sort_index()

    # Print the bin counts
    print(f"\n📊 Project counts by {bin_width}% coverage bins:")
    for label, count in bin_counts.items():
        print(f" - {label}: {count} project(s)")
    
    # Plot the histogram
    plt.figure(figsize=(10, 5))
    ax = bin_counts.plot(kind='bar', edgecolor='black')
    plt.title(f"Number of Projects by Image Coverage Bin ({bin_width}% bins)")
    plt.xlabel("Coverage Bin (%)")
    plt.ylabel("Number of Projects")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True)) # Force y-axis labels to be integers

    # Annotate the bars with their exact counts
    for i, count in enumerate(bin_counts):
        ax.annotate(str(count), xy=(i, count), xytext=(0,3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout
    plt.show()

def count_projs_wi_img_avail(df, threshold):
    """
    Counts and lists projects with image availability greater than or equal to the threshold.
    Returns a list of the project IDs with image availability >= to the threshold.
    
    Args:
        df (pd.DataFrame): Dataframe containing the image availability by project
        threshold (float): The minimum percent area covered required to include a project
    Returns:
        list: Project IDs meeting the threshold  
    """
    qualifying_projects = df[(df['total_percent_area_covered'] > threshold)]
    count = len(qualifying_projects)

    print(f"\n📊 Projects with ≥ {threshold}% image coverage: {count}")
    print("🆔 Project IDs:")
    for _, row in qualifying_projects.iterrows():
        pid = row['project_id']
        coverage = row['total_percent_area_covered']
        print(f" - {pid} → {coverage:.2f}%")
    return qualifying_projects['project_id'].tolist()

def count_total_low_covearage_polygons(df):
    print(f"📦 Total polygons with low imagery coverage: {len(df)}")

def count_with_images(df):
    has_image = df['num_images'] > 0
    count = has_image.sum()
    print(f"🛰️ Polygons with at least one image: {count} / {len(df)}")
    return df[has_image]

def count_with_multiple_images(df_with_images):
    multiple = df_with_images['num_images'] > 1
    count = multiple.sum()
    print(f"📸 Polygons with multiple available images: {count} / {len(df_with_images)}")
    return df_with_images

def explore_overlap_stats(df_with_images):
    print("\n🔍 Overlap Area (ha):")
    print(df_with_images['overlap_area_ha'].describe())

    print("\n🧮 Percent Image Coverage:")
    print(df_with_images['percent_img_cover'].describe())

def analyze_low_coverage_issues(df):
    print("\n--- Low Imagery Coverage Analysis ---")
    count_total_low_covearage_polygons(df)
    
    with_images_df = count_with_images(df)
    multiple_images_df = count_with_multiple_images(with_images_df)

    print("\n📊 Stats for all polygons with images:")
    explore_overlap_stats(with_images_df)

    print("\n📊 Stats for polygons with multiple images:")
    explore_overlap_stats(multiple_images_df)

def compare_polygon_coverage(baseline_df, ev_df, threshold):
    # Create dataframes with only relevent columns and rename for clarity before merging
    base = baseline_df[['poly_id', 'project_id', 'percent_img_cover']].rename(
        columns={'percent_img_cover': 'base_pct_img_cover'})
    ev = ev_df[['poly_id', 'percent_img_cover']].rename(
        columns={'percent_img_cover': 'ev_pct_img_cover'})
    
    # Merge dataframes on poly_id
    merged = base.merge(ev, on='poly_id', how='inner')

    # Filter polygons that meet the threshold in *both* periods
    merged['both_high'] = (
        (merged['base_pct_img_cover'] >= threshold) &
        (merged['ev_pct_img_cover'] >= threshold)
    )

    # Group by project and compute:
    # - total number of shared polygons
    # - number of polygons that meet threshold in both
    summary = (
        merged.groupby('project_id')
        .agg(total_polygons=('poly_id', 'count'),
             polygons_high_both=('both_high', 'sum'))
        .reset_index()
    )

    # Add percent
    summary['percent_polygons_high_both'] = (
        summary['polygons_high_both'] / summary['total_polygons'] * 100
    )

    return summary

### Functions to assess proportion of the cohort we can conduct tree count on now ###
# Percent of polygons with a valid plantstart date
def valid_plantstart_stats(df):
    grouped = df.groupby('project_id')['plantstart']
    count_valid = grouped.apply(lambda x: x.notna().sum()).rename('#_poly_valid_plantstart')
    total = grouped.size().rename('total_poly')
    pct_valid = (count_valid / total * 100).rename('%_poly_valid_plantstart')
    return pd.concat([total, pct_valid], axis=1).reset_index()

# Percent of polygons planted 2+ years ago
def planted_2_years_ago_stats(df, reference_date=None):
    if reference_date is None:
        reference_date = pd.Timestamp.today()
        print(reference_date)
    two_years_ago = reference_date - pd.DateOffset(years=2)

    grouped = df.groupby('project_id')['plantstart']
    count_2yrs = grouped.apply(lambda x: (x < two_years_ago).sum()).rename('#_poly_planted_2yr_ago')
    total = grouped.size().rename('total_poly')
    pct_2yrs = (count_2yrs / total * 100).rename('%_poly_planted_2yr_ago')
    return pd.concat([pct_2yrs], axis=1).reset_index()

# Percent of polygons with at least 1 filtered image 1 year+ post-plantstart
def ev_image_available_stats(df):
    grouped = df.groupby('project_id')['num_images']
    count_wi_img = grouped.apply(lambda x: (x > 0).sum()).rename('#_poly_ev_img')
    total = grouped.size().rename('total_poly')
    pct_wi_img = (count_wi_img / total * 100).rename('%_poly_ev_img')
    return pd.concat([pct_wi_img], axis=1).reset_index()

# Wrapper function
def summarize_project_planting_and_ev(df, reference_date=None):
    
    valid_start = valid_plantstart_stats(df)
    two_years = planted_2_years_ago_stats(df, reference_date)
    has_ev = ev_image_available_stats(df)

    # Merge all the summaries
    summary = valid_start.merge(two_years, on='project_id') \
                         .merge(has_ev, on='project_id')
    
    return summary