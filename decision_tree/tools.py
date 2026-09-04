import ast
import os

import pandas as pd
import yaml
from gri_shared_library.constants import TreeCountProjectPhaseDayRange

from gri_shared_library.os_tools import get_project_root_dir


def get_tm_auth_headers_from_secrets():
    # Set up token access
    auth_path = os.path.join(get_project_root_dir(), 'secrets.yaml')
    with open(auth_path) as auth_file:
        auth = yaml.safe_load(auth_file)
    headers = {
        'Authorization': f"Bearer {auth['tm_api']['tm_access_token']}"
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


def convert_to_os_path(target_dir, path_str):
    """
    Convert a given directory path string to a valid path format
    for the current operating system.
    """
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError("Path must be a non-empty string.")

    abs_path = os.path.join(get_project_root_dir(), target_dir, path_str)

    # Replace common wrong separators with OS-specific ones
    normalized_path = abs_path.replace("\\", os.sep).replace("/", os.sep)

    return normalized_path


def load_secrets(secrets_path):
    if os.path.isfile(secrets_path):
        with open(secrets_path, "r") as f:
            secrets_json = yaml.safe_load(f)
    else:
        # AWS
        aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID'] if 'AWS_ACCESS_KEY_ID' in os.environ else 'not_defined'
        aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY'] if 'AWS_SECRET_ACCESS_KEY' in os.environ else 'not_defined'
        aws_region = os.environ['AWS_REGION'] if 'AWS_REGION' in os.environ else 'us-east-1'
        aws = {"aws_access_key_id": aws_access_key_id, "aws_secret_access_key": aws_secret_access_key, "aws_region": aws_region}

        tm_access_token = os.environ['TM_ACCESS_TOKEN'] if 'TM_ACCESS_TOKEN' in os.environ else 'not_defined'
        tm_token = {"tm_access_token": tm_access_token}

        secrets_json = {
            "aws" : aws,
            "tm_api" : tm_token
        }

    return secrets_json


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_indicator_window_range(params, window_name):
    criteria = params.get('criteria', {})
    if window_name.lower() == 'baseline':
        baseline_range = criteria.get('baseline_range')
        if type(baseline_range) == str and baseline_range.lower() == 'default':
            return TreeCountProjectPhaseDayRange.BASELINE.value
        if type(baseline_range) == str and isinstance(ast.literal_eval(baseline_range), tuple):
            return ast.literal_eval(baseline_range)
        else:
            raise ValueError(f"Invalid baseline_range specification ({baseline_range}) in params file.")
    elif window_name.lower() == 'ext_baseline':
        ext_baseline_range = criteria.get('ext_baseline_range')
        if type(ext_baseline_range) == str and ext_baseline_range.lower() == 'default':
            return TreeCountProjectPhaseDayRange.EXT_BASELINE.value
        if type(ext_baseline_range) == str and isinstance(ast.literal_eval(ext_baseline_range), tuple):
            return ast.literal_eval(ext_baseline_range)
        else:
            raise ValueError(f"Invalid ext_baseline_range specification ({ext_baseline_range}) in params file.")
    elif window_name.lower() == 'early_insight':
        early_insight_range = criteria.get('ev_range')
        if type(early_insight_range) == str and early_insight_range.lower() == 'default':
            return TreeCountProjectPhaseDayRange.EARLY_INSIGHT.value
        if type(early_insight_range) == str and isinstance(ast.literal_eval(early_insight_range), tuple):
            return ast.literal_eval(early_insight_range)
        else:
            raise ValueError(f"Invalid early_insight_range specification ({early_insight_range}) in params file.")
    elif window_name.lower() == 'endline':
        endline_range = criteria.get('endline')
        if type(endline_range) == str and endline_range.lower() == 'default':
            return TreeCountProjectPhaseDayRange.ENDLINE.value
        if type(endline_range) == str and isinstance(ast.literal_eval(endline_range), tuple):
            return ast.literal_eval(endline_range)
        else:
            raise ValueError(f"Invalid endline_range specification ({endline_range}) in params file.")
    else:
        raise ValueError(f"Invalid window_name specification in params file.")


def append_note(df, idx, label, col='notes'):
    """
    Append `label` to the given notes column instead of overwriting whatever
    is already there. Notes accumulate as a list (e.g. 'missing-plantstart; 
    ttc-bad-year'); existing labels are not duplicated.

    Parameters:
    - df (pd.DataFrame): DataFrame with the target notes column.
    - idx: either a single row index (updates df.at[idx, col]) or a
      boolean mask aligned to df.index (vectorized update for many rows).
    - label (str): note label to append, e.g. 'missing-ttc', 'ttc-bad-year'.
    - col (str): which notes column to update, e.g. 'notes_base', 'notes_ev'.
      Defaults to 'notes' for backward compatibility.
    """
    def _merge(current):
        if pd.isna(current) or current == '':
            return label
        parts = [p.strip() for p in str(current).split(';')]
        return current if label in parts else f"{current}; {label}"

    if isinstance(idx, pd.Series):
        # boolean mask - vectorized update across matching rows
        if not idx.any():
            return
        df.loc[idx, col] = df.loc[idx, col].apply(_merge)
    else:
        # single row index
        df.at[idx, col] = _merge(df.at[idx, col])


def place_column_after(df, col, anchor):
    """
    Move `col` to sit immediately after `anchor` in df's column order.
    No-op if either column is missing (e.g. an intermediate/test df that
    doesn't have the column yet).
    """
    if col not in df.columns or anchor not in df.columns:
        return df
    series = df.pop(col)
    loc = df.columns.get_loc(anchor) + 1
    df.insert(loc, col, series)
    return df
