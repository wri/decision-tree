import os
import sys

import yaml
from pathlib import Path


def convert_to_os_path(target_dir, path_str):
    """
    Convert a given directory path string to a valid path format
    for the current operating system.
    """
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError("Path must be a non-empty string.")

    abs_path = os.path.join(get_project_root(), target_dir, path_str)

    # Replace common wrong separators with OS-specific ones
    normalized_path = abs_path.replace("\\", os.sep).replace("/", os.sep)

    return normalized_path

def load_secrets(secrets_path):
    if os.path.isfile(secrets_path):
        with open(secrets_path, "r") as f:
            secrets_json = yaml.safe_load(f)
    else:
        # Open Topology
        opentopo_api_key = os.environ['OPENTOPO_API_KEY'] if 'OPENTOPO_API_KEY' in os.environ else 'not_defined'
        open_topo = {"opentopo_api_key": opentopo_api_key}

        # AWS
        aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'] if 'AWS_ACCESS_KEY_ID' in os.environ else 'not_defined'
        aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY'] if 'AWS_SECRET_ACCESS_KEY' in os.environ else 'not_defined'
        aws_region = os.environ['AWS_REGION'] if 'AWS_REGION' in os.environ else 'us-east-1'
        aws = {"aws_access_key_id": aws_access_key_id, "aws_secret_access_key": aws_secret_access_key, "aws_region": aws_region}

        asana_pat = os.environ['ASANA_PAT'] if 'ASANA_PAT' in os.environ else 'not_defined'
        asana = {"asana_pat": asana_pat}

        gfw_access_token = os.environ['GFW_ACCESS_TOKEN'] if 'GFW_ACCESS_TOKEN' in os.environ else 'not_defined'
        gfw = {"gfw_access_token": gfw_access_token}

        secrets_json = {
            "opentopo" : open_topo,
            "aws" : aws,
            "asana" : asana,
            "gfw" : gfw
        }

    return secrets_json


def get_project_root(start_path=None, markers=None):
    """
    Get the root directory of a project by searching for marker files.

    Args:
        start_path (str, optional): Directory to start searching from.
            Defaults to current working directory.
        markers (list of str, optional): Files or directories that indicate the root.
            Defaults to ['pyproject.toml', 'setup.py', '.git'].

    Returns:
        str: Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If no root directory found.
    """
    if start_path is None:
        start_path = os.getcwd()

    if markers is None:
        markers = ['pyproject.toml', 'setup.py', '.git']

    current_path = os.path.abspath(start_path)

    while True:
        if any(os.path.exists(os.path.join(current_path, marker)) for marker in markers):
            return current_path

        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            # Reached root of filesystem without finding markers
            raise FileNotFoundError(
                f"Could not find project root starting from {start_path} "
                f"using markers: {markers}"
            )
        current_path = parent_path
