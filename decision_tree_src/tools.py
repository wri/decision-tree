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


def get_project_root(marker_files=("pyproject.toml", "requirements.txt", ".git")) -> str:
    """Return the absolute path to the project root directory."""
    # Start from the directory of the current file
    # return str(Path(os.path.dirname(__file__)).parent)
    # Get the path of the script that started execution
    if hasattr(sys.modules['__main__'], '__file__'):
        start_path = Path(sys.modules['__main__'].__file__).resolve()
    else:
        # Fallback for interactive sessions
        start_path = Path.cwd()

    # Walk upward until we find a marker
    for parent in [start_path] + list(start_path.parents):
        for marker in marker_files:
            if (parent / marker).exists():
                return str(parent)

    raise FileNotFoundError(
        f"Could not find project root. None of {marker_files} found above {start_path}"
    )