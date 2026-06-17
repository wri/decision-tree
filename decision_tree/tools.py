import os
import yaml

from gri_shared_library.os_tools import get_project_root_dir


def get_tm_auth_headers_from_secrets():
    # Set up token access
    auth_path = os.path.join(get_project_root_dir(), 'secrets.yaml')
    with open(auth_path) as auth_file:
        auth = yaml.safe_load(auth_file)
    headers = {
        'Authorization': f"Bearer {auth['tm_api_auth']['tm_access_token']}"
        }
    return headers


def get_tm_auth():
    if 'TM_ACCESS_TOKEN' in os.environ:
        tm_access_token = os.environ['TM_ACCESS_TOKEN']
        auth_headers = {
            'Authorization': f"Bearer {tm_access_token}"
        }
    else:
        auth_headers = get_tm_auth_headers_from_secrets()

    return auth_headers


# def convert_to_os_path(target_dir, path_str):
#     """
#     Convert a given directory path string to a valid path format
#     for the current operating system.
#     """
#     if not isinstance(path_str, str) or not path_str.strip():
#         raise ValueError("Path must be a non-empty string.")
#
#     abs_path = os.path.join(get_project_root_dir(), target_dir, path_str)
#
#     # Replace common wrong separators with OS-specific ones
#     normalized_path = abs_path.replace("\\", os.sep).replace("/", os.sep)
#
#     return normalized_path
#
#
# def get_gfw_access_token(params):
#     if 'GFW_ACCESS_TOKEN' in os.environ:
#         access_token = os.environ['GFW_ACCESS_TOKEN']
#         auth_headers = {
#             'Authorization': f"Bearer {access_token}"
#         }
#     else:
#         tm_auth_file = params['config']
#         tm_auth_path = os.path.join(get_project_root_dir(), tm_auth_file)
#         with open(tm_auth_path) as auth_file:
#             auth = yaml.safe_load(auth_file)
#         access_token = auth['gfw_api']['access_token']
#         auth_headers = {'Authorization': f"Bearer {access_token}"}
#
#     return auth_headers
#
#
# def load_secrets(secrets_path):
#     if os.path.isfile(secrets_path):
#         with open(secrets_path, "r") as f:
#             secrets_json = yaml.safe_load(f)
#     else:
#         # Open Topology
#         opentopo_api_key = os.environ['OPENTOPO_API_KEY'] if 'OPENTOPO_API_KEY' in os.environ else 'not_defined'
#         open_topo = {"opentopo_api_key": opentopo_api_key}
#
#         # AWS
#         aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'] if 'AWS_ACCESS_KEY_ID' in os.environ else 'not_defined'
#         aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY'] if 'AWS_SECRET_ACCESS_KEY' in os.environ else 'not_defined'
#         aws_region = os.environ['AWS_REGION'] if 'AWS_REGION' in os.environ else 'us-east-1'
#         aws = {"aws_access_key_id": aws_access_key_id, "aws_secret_access_key": aws_secret_access_key, "aws_region": aws_region}
#
#         asana_pat = os.environ['ASANA_PAT'] if 'ASANA_PAT' in os.environ else 'not_defined'
#         asana = {"asana_pat": asana_pat}
#
#         tm_access_token = os.environ['TM_ACCESS_TOKEN'] if 'TM_ACCESS_TOKEN' in os.environ else 'not_defined'
#         tm_token = {"tm_access_token": tm_access_token}
#
#         secrets_json = {
#             "opentopo" : open_topo,
#             "aws" : aws,
#             "asana" : asana,
#             "tm_api" : tm_token
#         }
#
#     return secrets_json
#
