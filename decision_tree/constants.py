from datetime import date
from enum import Enum

from gri_shared_library.os_tools import get_project_root_dir


# Construct list of desired columns
TF_START_YR = 2020
current_year = date.today().year
ttc_years = []
for year in range(TF_START_YR, current_year, 1):
    ttc_year = f"ttc_{year}"
    ttc_years.append(ttc_year)

fixed_cols = [
    'project_id', 'poly_id', 'site_id', 'project_name',
    'plantstart', 'practice', 'target_sys', 'baseline_year', 'ev_year',
    'area',
    'baseline_img_count', 'ev_img_count', 'baseline_canopy',
    'ev_canopy', 'slope_area', 'slope', 'baseline_decision'
]

DESIRED_COLS = fixed_cols + ttc_years

TM_STAGING_URI = 'https://api-staging.terramatch.org/research/v3/sitePolygons?' # not currently used
TM_PROD_URI = 'https://api.terramatch.org/research/v3/sitePolygons?'
PROJECT_ROOT = get_project_root_dir()
OPENTOPO_URI = 'https://portal.opentopography.org/API/globaldem'

RULES = 'rule_template.csv'

COST_FIELD = 20.0
COST_REMOTE = 0.32

ASANA_GID = 1209713669431043

class TestProjectHandling(Enum):
    INCLUDE = "include"   # test + non-test
    EXCLUDE = "exclude"   # non-test only
    ONLY = "only"         # test only
