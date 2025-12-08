import os
from pathlib import Path

import yaml


def get_gfw_access_token(params):
    if 'GFW_ACCESS_TOKEN' in os.environ:
        access_token = os.environ['GFW_ACCESS_TOKEN']
        auth_headers = {
            'Authorization': f"Bearer {access_token}"
        }
    else:
        tm_auth_file = params['config']
        tm_auth_path = os.path.join(get_project_root(), tm_auth_file)
        with open(tm_auth_path) as auth_file:
            auth = yaml.safe_load(auth_file)
        auth_headers = {'Authorization': f"Bearer {auth['access_token']}"}

    return auth_headers


def read_opentopo_api_key(config):
    if 'GFW_ACCESS_TOKEN' in os.environ:
        api_key = os.environ['OPENTOPO_API_KEY']
    else:
        api_key = config['opentopo_key']

    return api_key


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    # Start from the directory of the current file
    return str(Path(os.path.dirname(__file__)).parent)
