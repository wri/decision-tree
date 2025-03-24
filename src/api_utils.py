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



## legacy wrapper function, use above ^^
def legacy(url, headers, project_ids, outfile="../data/tm_api_response.json"):
    '''
    edits to legacy func (below):
        - params are defined within the func to facilitate looping through
        various project ids -- line 23
        - adds tqdm progress print out bar (not required but helpful!) -- line 20
        - defines new_last_record variable as none -- line 31
        - dumps json to outfile, added as func arg -- line 67
        - adds project id to output support subsequent maxar API request -- line 51      
    '''
    results = []
    with tqdm(total=len(project_ids), desc="Processing Projects", unit="project") as progress_bar:
        for project_id in project_ids:
            # Set parameters with the current project ID
            params = {
                'projectId[]': project_id,
                'polygonStatus[]': 'approved',
                'includeTestProjects': 'false',
                'page[size]': '100'
            }

            last_record = ''
            new_last_record = None  # Ensure it's defined before use

            while True:
                # Send request
                response = requests.get(url, headers=headers, params=params)

                # Check status code
                if response.status_code != 200:
                    raise ValueError(f'Request failed for project {project_id} with status code {response.status_code}')
                
                response_json = response.json()
                total_records = response_json['meta']['page']['total']

                # Parse response data
                if total_records == 0:
                    break  # Exit if no data is available

                for idx in range(total_records):
                    data = response_json['data'][idx]['attributes']
                    data['poly_id'] = response_json['data'][idx]['meta']['page']['cursor']
                    data['project_id'] = project_id 
                    results.append(data)

                    # Assign the last cursor only if there are records
                    if idx == (total_records - 1):
                        new_last_record = response_json['data'][idx]['meta']['page']['cursor']

                # Check if there are more pages
                if new_last_record and last_record != new_last_record:
                    last_record = new_last_record
                    params['page[after]'] = last_record
                else:
                    break  # Exit pagination if no new cursor is found

            progress_bar.update(1) 
            
    with open(outfile, "w") as file:
        json.dump(results, file, indent=4)

    return results