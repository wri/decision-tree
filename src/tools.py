import os
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
        auth_headers = {'Authorization': f"Bearer {auth['access_token']}"}

    return auth_headers


def get_opentopo_api_key(config):
    if 'GFW_ACCESS_TOKEN' in os.environ:
        api_key = os.environ['OPENTOPO_API_KEY']
    else:
        api_key = config['opentopo_key']

    return api_key


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    # Start from the directory of the current file
    return str(Path(os.path.dirname(__file__)).parent)
