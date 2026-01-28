import os
import sys

import yaml
from pathlib import Path


def convert_to_os_path(path_str):
    """
    Convert a given directory path string to a valid path format
    for the current operating system.

    Args:
        path_str (str): The input path string (any format).

    Returns:
        str: A normalized, OS-valid path string.
    """
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError("Path must be a non-empty string.")

    # Replace common wrong separators with OS-specific ones
    normalized_path = path_str.replace("\\", os.sep).replace("/", os.sep)

    # Use pathlib to resolve and normalize
    try:
        path_obj = Path(normalized_path).expanduser().resolve(strict=False)
    except Exception as e:
        raise ValueError(f"Invalid path format: {e}")

    return str(path_obj)


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
        access_token = auth['gfw_api']['access_token']
        auth_headers = {'Authorization': f"Bearer {access_token}"}

    return auth_headers


def get_opentopo_api_key(config):
    if 'GFW_ACCESS_TOKEN' in os.environ:
        api_key = os.environ['OPENTOPO_API_KEY']
    else:
        api_key = config['opentopo_key']

    return api_key


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
