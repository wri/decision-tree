import requests
import yaml
import json
import pandas as pd
import json
from tqdm import tqdm
from tm_api_utils import pull_tm_api_data


def pull_wrapper(url, 
                headers, 
                project_ids, 
                outfile=None,
                ):
    """
    Wrapper function around the TM API package.
    
    Iterates over a list of project IDs, aggregates results, and writes to JSON if specified.

    Parameters:
        url (str): TerraMatch API endpoint.
        headers (dict): Auth headers.
        project_ids (list): List of project UUIDs.
        outfile (str): Optional path to write output JSON.
        show_progress (bool): Whether to show tqdm progress bar.
    """
    all_results = []

    progress = tqdm(total=len(project_ids), desc="Pulling Projects", unit="project")

    for project_id in project_ids:
        params = {
            'projectId[]': project_id,
            'polygonStatus[]': 'approved',
            'includeTestProjects': 'false',
            'page[size]': '100'
        }

        try:
            results = pull_tm_api_data(url, headers, params)
            if results is None:
                print(f"No results returned for project: {project_id}")
                continue
            
            # Adds prj ID for traceability
            for r in results:
                r['project_id'] = project_id  
            all_results.extend(results)

        except Exception as e:
            print(f"Error pulling project {project_id}: {e}")
        
        progress.update(1)

    progress.close()

    # Optional: Write to JSON file
    if outfile:
        with open(outfile, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {outfile}")

    return all_results