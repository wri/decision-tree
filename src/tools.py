import os

import yaml


def get_gfw_access_token(params):
    if 'GFW_ACCESS_TOKEN' in os.environ:
        access_token = os.environ['GFW_ACCESS_TOKEN']
        auth_headers = {
            'Authorization': f"Bearer {access_token}"
        }
    else:
        tm_auth_path = params['config']
        with open(tm_auth_path) as auth_file:
            auth = yaml.safe_load(auth_file)
        auth_headers = {'Authorization': f"Bearer {auth['access_token']}"}

    return auth_headers
