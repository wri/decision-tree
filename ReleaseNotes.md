# Release notes for decision-tree

## 2026/03/09
1. Restructured secrets file for opentopo, asana, and gfw. See secrets_template.yaml file for updated format.
2. Enabled tests for all options using test project
3. params.yaml - Removed URL specifications from params.yaml file and moved into constants.py file.
4. params.yaml - Added tm_environment option to params.yaml file to specify whether to use prod or staging URL
5. Added temporary caching of DEM files from OpenTopo.
6. Began using basic functions in the gri-pipeline shared_library.
7. Tweaked the toml and setup_uv.sh files based on current understanding of what is needed for both linux and macos

## 2026/01/30
1. Resolved relative paths to absolute paths

## 2026/01/21
1. modified run_decision_tree.py to include two arguments on the main function, thereby allowing both manual and programmatic execution.
2. added testing for functionality in run_decision_tree.py
3. three tests are currently commented out while we await synthetic data for imaginary projects
4. resolves a number of bugs which may have emerged due to use of newer versions of dependent packages.
5. Updated code per discussion with Jessica for PR