import os
from pathlib import Path

import yaml


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    # Start from the directory of the current file
    return str(Path(os.path.dirname(__file__)).parent)

def get_opentopo_api_key(params):
    if 'OPENTOPO_API_KEY' in os.environ:
        api_key = os.environ['OPENTOPO_API_KEY']
    else:
        secrets_yml = params['config']
        with open(secrets_yml) as auth_file:
            auth = yaml.safe_load(auth_file)
        api_key = auth['opentopo_key']

    return api_key
