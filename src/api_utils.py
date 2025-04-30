import requests
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


def pull_wrapper(url, headers, project_ids, outfile=None):
    """
    Wrapper function around the TM API package.

    Iterates over a list of project IDs, aggregates results, and writes to JSON if specified.

    Parameters:
        url (str): TerraMatch API endpoint.
        headers (dict): Auth headers.
        project_ids (list): List of project UUIDs.
        outfile (str): Optional path to write output JSON.
    """
    all_results = []

    with tqdm(total=len(project_ids), desc="Pulling Projects", unit="project") as progress:
        for project_id in project_ids:
            params = {
                'projectId[]': project_id,
                'polygonStatus[]': 'approved',
                'includeTestProjects': 'false',
                'page[size]': '100'
            }

            try:
                results = patched_pull_tm_api_data(url, headers, params)

                if results is None:
                    print(f"No results returned for project: {project_id}")
                    continue

                for r in results:
                    r['project_id'] = project_id  # Add project_id for traceability

                all_results.extend(results)

            except Exception as e:
                print(f"Error pulling project {project_id}: {e}")

            progress.update(1)

    # Optional: write to JSON file
    if outfile:
        with open(outfile, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved to {outfile}")

    return all_results