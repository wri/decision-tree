import yaml
import sys
sys.path.append('../src/')


import process_api_results as clean
import api_utils as tm
from tm_api_utils import pull_tm_api_data

import image_availability as img
import canopy_cover as cover

import decision_trees as tree

# set up required inputs
yaml_path = "params.yaml"
with open(yaml_path, "r") as f:
    params = yaml.safe_load(f)

tm_auth_path = params['tm_api']['tm_auth_path']
with open(tm_auth_path) as auth_file:
    auth = yaml.safe_load(auth_file)

headers = {
    'Authorization': f"Bearer {auth['access_token']}"
    }

out = params['outfile']
today = out['today']
tm_response = out['tm_response'].format(cohort=out['cohort'],today=out['today'])
feats = out['feats'].format(cohort=out['cohort'],today=out['today'])
feats_maxar_query = out['feats_maxar_query'].format(cohort=out['cohort'],today=out['today'])
maxar_meta = out['maxar_meta'].format(cohort=out['cohort'],today=out['today'])
results = out['decision'].format(cohort=out['cohort'],today=out['today'])


json_response = tm.pull_wrapper(params['tm_api']['tm_prod_url'], 
                             headers, 
                             c1_ids, # need to identify better way to get ids
                             tm_response)

response_clean = clean.process_tm_api_results(json_response, # has to read this file in?
                                              params['criteria']['drop_missing'],
                                              outfile1=feats, 
                                              outfile2=feats_maxar_query)

# step to acquire / process slope data goes here

img_availability = img.analyze_image_availability(response_clean, # or read in from csv 
                                                   maxar_meta, # this is a string 
                                                   tuple(params['criteria']['baseline_range']), 
                                                   tuple(params['criteria']['ev_range']),
                                                   params['criteria']['cloud_thresh'])

canopy_cover = cover.apply_canopy_classification(img_availability,
                                                params['criteria']['canopy_threshold'],
                                                tuple(params['criteria']['baseline_range']), 
                                                tuple(params['criteria']['ev_range']))
# slope branch would go here

c1_final = tree.apply_rules_baseline(canopy_cover, 
                                    params['criteria']['rules'],
                                    save_to_csv=results)

def summarize_results(df):
    total_projects = df['project_id'].nunique()
    print(f"{total_projects} total projects")
    
    # 2. Total number of polygon_ids per project
    polygon_counts = df.groupby('project_id')['poly_id'].nunique().reset_index(name='polygon_count')
    
    # 3. Proportion of remote vs field decisions per project
    decision_counts = df.groupby(['project_id', 'decision']).size().unstack(fill_value=0)    
    decision_proportions = decision_counts.div(decision_counts.sum(axis=1), axis=0)
    decision_proportions = (decision_proportions * 100).round(2).reset_index()
    
    # Merge polygon counts and decision proportions into one summary
    summary = polygon_counts.merge(decision_proportions, on='project_id')
    return summary

test = summarize_results(c1_final)
