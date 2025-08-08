import requests
import certifi
import yaml
import json
import pandas as pd
import json
from tqdm import tqdm
from tm_api_utils import pull_tm_api_data

def patched_pull_tm_api_data(url: str, headers: dict, params: dict) -> list:
    """
    ## REMOVE ME ONCE PACKAGE UPDATED ##
    
    Optimized version of patched_pull_tm_api_data with improved performance.

    - Parses response JSON only once per request
    - Efficiently handles pagination cursors
    - Uses safer access to nested metadata
    """
    results = []
    last_record = None

    while True:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            raise ValueError(f"Request failed: {response.status_code}")

        response_json = response.json()
        data_items = response_json.get('data', [])

        if not data_items:
            break

        for item in data_items:
            attributes = item.get('attributes', {})
            attributes['poly_id'] = item.get('meta', {}).get('page', {}).get('cursor') or item.get('id')
            results.append(attributes)

        # Efficient cursor handling after processing all records
        new_last_record = data_items[-1].get('meta', {}).get('page', {}).get('cursor') if data_items else None

        if new_last_record and new_last_record != last_record:
            last_record = new_last_record
            params['page[after]'] = last_record
        else:
            break

    return results

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
        #print(params)

        try:
            #results = pull_tm_api_data(url, headers, params)
            results = patched_pull_tm_api_data(url, headers, params) # Trying the new function found on https://github.com/wri/terrafund-portfolio-analyses/blob/main/src/api_utils.py
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

DEFAULT_INDICATOR_MAPPING = {
    'treeCover': 'tree_cover',
    'treeCoverLoss': 'tree_cover_loss',
    'treeCoverLossFires': 'tree_cover_loss_fires',
    'restorationByStrategy': 'restoration_by_strategy',
    'restorationByLandUse': 'restoration_by_land_use',
    'restorationByEcoRegion': 'restoration_by_ecoregion'
}

def parse_tm_api_results(results, outfile, parse_indicators=False, indicator_mapping=None):
    """
    Converts TerraMatch API results JSON into a structured DataFrame with selected fields.
    
    Args:
        results (list): Raw JSON results from the API (list of dicts)
        outfile (str): Path to save cleaned CSV output
        parse_indicators (bool): Include indicators columns in the final DataFrame
        indicator_mapping (dict[str, str]): Dictionary used to map indicatorSlug names to desired column names. Keys should be the indicatorSlug keys within the results dictionary. 
          Values should be the desired column name in the final DataFrame. If parse_indicators = True, this defaults to the DEFAULT_INDICATOR_MAPPING. 

    Returns:
        final_df (pd.DataFrame): Structured dataframe with selected fields 
    """
    extracted_data = []

    # Iterate over each feature in the results JSON to extract polygon information
    for feature in results: 
        # Basic attributes
        row_data = {
            'project_id': feature.get('project_id'),
            'project_name': feature.get('projectShortName'),
            'poly_id': feature.get('poly_id'),
            'site_id': feature.get('siteId'),
            'geometry': feature.get('geometry'),
            'plantstart': feature.get('plantStart'),
            'plantend': feature.get('plantEnd'),
            'practice': feature.get('practice'),
            'target_sys': feature.get('targetSys'),
            'dist': feature.get('distr'),
            'project_phase': feature.get('projectPhase', '')  # default if missing
        }

        # Optionally parse the 'indicators' list into separate columns
        if parse_indicators:
            if indicator_mapping is None:
                indicator_mapping = DEFAULT_INDICATOR_MAPPING
            elif not isinstance(indicator_mapping, dict):
                raise ValueError("indicator_mapping must be provided as a dictionary.")
            
            # Get the value associated with the 'indicators' key
            indicators = feature.get('indicators', [])
            # For each indicator dicationary
            for indicator in indicators:
                slug = indicator.get('indicatorSlug')
                if slug in indicator_mapping:
                    col_name = indicator_mapping[slug]
                    row_data[col_name] = indicator  # Keep full dictionary

        extracted_data.append(row_data)

    final_df = pd.DataFrame(extracted_data)
    
    # Save results
    final_df.to_csv(outfile, index=False)

    return final_df
