import json
import os
import shutil
import yaml

from pathlib import Path
from src.api_utils import tm_pull_wrapper
from tests.tools import get_project_root

ROOT_PATH = get_project_root()

def test_tm_pull_wrapper():
    parms_path = os.path.join(ROOT_PATH, "tests", "data", "params.yaml")
    with open(parms_path, 'r') as file:
        params = yaml.safe_load(file)

    # Determine outfile path
    out = params['outfile']
    data_version = out['data_version']
    outfile = out['tm_response'].format(cohort=out['cohort'], data_version=data_version)

    # Prepare output folder
    out_directory = Path(outfile).parent
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    else:
        if os.path.isfile(outfile):
            shutil.rmtree(outfile)

    ids = ['468bee12-bfbc-4387-a00a-d7e915576427']
    tm_pull_wrapper(params, ids)

    # Confirm that the output file was generated
    assert os.path.isfile(outfile)

    # Confirm that the file contains an expected number of polygons or more
    expected_minimum_poly_count = 24
    with open(outfile, 'r') as file:
        data = json.load(file)
    assert len(data) >= expected_minimum_poly_count

    # Cleanup
    shutil.rmtree(out_directory)

