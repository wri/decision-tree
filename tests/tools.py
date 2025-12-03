import os

import yaml

from src.tools import get_project_root


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
