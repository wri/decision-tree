import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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