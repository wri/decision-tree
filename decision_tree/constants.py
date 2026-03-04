from datetime import date

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
