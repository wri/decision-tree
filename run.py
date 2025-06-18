import yaml
import sys
import pandas as pd
sys.path.append('../src/')

import process_api_results as clean
import api_utils as tm
from tm_api_utils import pull_tm_api_data

import image_availability as img
import canopy_cover as cover
import slope

import decision_trees as tree

# set up required inputs
yaml_path = "params.yaml"
with open(yaml_path, "r") as f:
    params = yaml.safe_load(f)

config_path = "../secrets.yaml"
with open(config_path) as conf_file:
    config = yaml.safe_load(conf_file)


# these need to be defined here becuase they use string formatting
out = params['outfile']
today = out['today']
tm_outfile = out['tm_response'].format(cohort=out['cohort'],today=out['today'])
ot_outfile = out['ot_response'].format(cohort=out['cohort'],today=out['today'])
feats = out['feats'].format(cohort=out['cohort'],today=out['today'])
feats_maxar_query = out['feats_maxar_query'].format(cohort=out['cohort'],today=out['today'])
maxar_meta = out['maxar_meta'].format(cohort=out['cohort'],today=out['today'])
results = out['decision'].format(cohort=out['cohort'],today=out['today'])
geojson_dir = out['geojsons']

# where should i define the parameters? filenames defined above bc of string
# formatting and variables defined within the functions?
# should each step reference the prior variable or the locally saved file?

def get_ids(params):
    out = params['outfile']
    portfolio = out['portfolio']
    full = pd.read_csv(portfolio)
    cohort = out['cohort']
    keyword = 'terrafund' if cohort == 'c1' else 'terrafund-landscapes'
    filtered = full[(full.cohort == keyword)]
    ids = list(set(filtered.project_id))
    return ids
ids = get_ids(params)

tm_response = tm.tm_pull_wrapper(params, 
                             ids, # need to identify better way to get ids
                             today,
                             tm_outfile,
                             geojson_dir)

tm_clean = clean.process_tm_api_results(tm_response, # has to read this file in?
                                        params['criteria']['drop_missing'],
                                        outfile1=feats, 
                                        outfile2=feats_maxar_query)

ot_response = tm.opentopo_pull_wrapper(params['opt_api']['opt_url'],
                                       config['opentopo_key'],
                                       tm_clean,
                                       geojson_dir,
                                       params['criteria']['slope_thresh'],
                                       outfile=ot_outfile)

img_branch = img.analyze_image_availability(tm_clean, # or read in from csv 
                                            maxar_meta, # this is a string 
                                            params)

canopy_branch = cover.apply_canopy_classification(img_branch, params)

slope_branch = slope.calculate_slope()

c1_final = tree.apply_rules_baseline(canopy_branch, 
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
