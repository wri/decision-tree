from datetime import date

from gri_shared_library.os_tools import get_project_root_dir


# Construct list of desired columns
start_year = 2020
current_year = date.today().year
ttc_years = []
for year in range(start_year, current_year, 1):
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


BASE_OFFSET_YRS = 1
EI_OFFSET_YRS = 2

COST_FIELD = 20.0
COST_REMOTE = 0.32