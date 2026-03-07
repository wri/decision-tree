import os
import shutil

import yaml

from decision_tree.tools import get_project_root

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


def delete_file(folder, filename):
    scratch_file_path1 = os.path.join(ROOT_PATH, "tests", "data", folder, filename)
    if os.path.isfile(scratch_file_path1):
        os.remove(scratch_file_path1)

def delete_folder(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)

def delete_source_geojsons_file(filename):
    scratch_file_path1 = os.path.join(ROOT_PATH, "tests", "data", "geojsons", filename)
    if os.path.isfile(scratch_file_path1):
        os.remove(scratch_file_path1)
