import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import math


def method_piechart_perprj(df):
    """
    Iterates through each unique project in the 'project' column and creates a pie chart 
    illustrating the percentage breakdown of 'method' for each project.

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns 'project' and 'method'.

    Returns:
    None: Displays pie charts for each project.
    """
    # Get the unique projects
    projects = df['project'].unique()
    num_projects = len(projects)
    
    # Calculate the number of rows and columns for subplots
    cols = math.ceil(math.sqrt(num_projects))
    rows = math.ceil(num_projects / cols)

    # Define a consistent color scheme for the methods
    color_dict = {
        'remote': '#66b3ff',  # Blue for remote
        'field': '#99ff99'    # Green for field
    }

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Flatten axes for easy iteration, in case of a single row or column
    axes = axes.flatten()
    
    # Loop through each project and plot
    for i, project in enumerate(projects):
        # Filter the dataframe for the current project
        project_df = df[df['project'] == project]

        # Get the method breakdown
        method_counts = project_df['method'].value_counts()

        # Create the color list based on the method counts
        colors = [color_dict[method] for method in method_counts.index]

        # Custom label function to show both count and percentage
        def autopct_format(pct):
            total = sum(method_counts)
            count = int(round(pct * total / 100.0))
            return f'{count} ({pct:.1f}%)'

        # Plot the pie chart in the respective subplot
        wedges, texts, autotexts = axes[i].pie(
            method_counts, 
            labels=method_counts.index, 
            autopct=autopct_format, 
            startangle=90, 
            colors=colors
        )
        
        axes[i].set_title(f"{project}", fontsize=12)

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Show the plots
    plt.show()
