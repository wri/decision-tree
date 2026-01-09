import os
import yaml

from src.tools import get_project_root

ROOT_PATH = get_project_root()

def get_opentopo_api_key(params):
    if 'OPENTOPO_API_KEY' in os.environ:
        api_key = os.environ['OPENTOPO_API_KEY']
    else:
        tm_auth_file = params['config']
        tm_auth_path = os.path.join(get_project_root(), tm_auth_file)
        with open(tm_auth_path) as auth_file:
            auth = yaml.safe_load(auth_file)
        api_key = auth['opentopo_key']

    return api_key

def standardize_test_param_paths(params):
    root_path = get_project_root()
    params['criteria']['rules'] = os.path.join(root_path, "tests", params['criteria']['rules'])
    params['outfile']['portfolio'] = os.path.join(root_path, "tests", params['outfile']['portfolio'])
    params['outfile']['geojsons'] = os.path.join(root_path, "tests", params['outfile']['geojsons'])
    params['outfile']['tm_response'] = os.path.join(root_path, "tests", params['outfile']['tm_response'])
    params['outfile']['slope_stats'] = os.path.join(root_path, "tests", params['outfile']['slope_stats'])
    params['outfile']['project_stats'] = os.path.join(root_path, "tests", params['outfile']['project_stats'])
    params['outfile']['feats'] = os.path.join(root_path, "tests", params['outfile']['feats'])
    params['outfile']['feats_maxar'] = os.path.join(root_path, "tests", params['outfile']['feats_maxar'])
    params['outfile']['maxar_meta'] = os.path.join(root_path, "tests", params['outfile']['maxar_meta'])
    params['outfile']['tree_results'] = os.path.join(root_path, "tests", params['outfile']['tree_results'])
    params['outfile']['poly_decision'] = os.path.join(root_path, "tests", params['outfile']['poly_decision'])
    params['outfile']['prj_decision'] = os.path.join(root_path, "tests", params['outfile']['prj_decision'])

    return params


def delete_scratch_file(filename):
    scratch_file_path1 = os.path.join(ROOT_PATH, "tests", "data", "scratch_files", filename)
    if os.path.isfile(scratch_file_path1):
        os.remove(scratch_file_path1)


def delete_source_geojsons_file(filename):
    scratch_file_path1 = os.path.join(ROOT_PATH, "tests", "data", "source_geojsons", filename)
    if os.path.isfile(scratch_file_path1):
        os.remove(scratch_file_path1)
