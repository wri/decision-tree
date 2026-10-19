# Release notes for decision-tree

## 2026/10/09
1. Added run_polygon_metrics and run_project_metrics methods and removed run_decision_tree function.
2. run_polygon_metrics always saves poly_results to a local csv file.
3. Updated tests for modified functions.
4. Added other unit tests
5. Added rds_dao.py for saving results to and reading them from RDS via an SSM tunnel through the bastion (requires AWS CLI v2, Session Manager plugin, psycopg).
6. params.yaml
   1. Added rds section; rds.upload controls saving results to RDS (replaces s3.upload).
   2. Added outfile.save_local_intermediates (replaces the checkpoint argument and --checkpoint option).
   3. Added aws.bastion_id_for_rds; RDS uses aws.land_aws_profile and its region.
7. run_project_metrics requires project_ids; if poly_results is not supplied, the latest polygons are read from RDS. 
8. All rows from one run share a single run_datetime. 
9. Added RDS unit and integration tests (integration tests delete their RDS rows); moved test_rds_query.py to integration; re-enabled test_run_decision_tree_full.

## 2026/10/08
1. Switched dependency from the deprecated terramatch-researcher-api repo to shared-library repo

## 2026/10/07
1. Split tests into folders for unit and integration
2. Modified workflow to only execute unit tests

## 2026/08/28
1. Removed used of Asana

## 2026/08/27
1. Added the notes_base and notes_ev columns to the poly_results dataframe for recording additional context as to why decisions were made.

## 2026/08/06
1. Removed OpenTopo references
2. Improved performance of slope computation
3. Updated S3 credential handling to use AWS-profile specified in secrets.yaml

## 2026/06/25
1. Converted indicator-windows to use enum definitions in gri-shared-library

## 2026/06/10
1. Re-added option to read TerraMatch polygon data through the TerraMatch API.
2. Added "tm_source" option to params.yaml files for controlling whether TM data are read from the geoparquet file or the TM API.
3. Fixed bug in reading cohort from geoparquet.

## 2026/04/02
1. Replaced TerraMatch API-based data ingestion with a GeoParquet-based approach.
2. Refactors the pipeline inputs / outputs by adding optional checkpoints to separate between research/production mode.

## 2026/03/26
1. Switched to using gri-shared-library repo

## 2026/03/09
1. Restructured secrets file for opentopo, asana, and gfw. See secrets_template.yaml file for updated format.
2. Enabled tests for all options using test project
3. params.yaml - Removed URL specifications from params.yaml file and moved into constants.py file.
4. params.yaml - Added tm_environment option to params.yaml file to specify whether to use prod or staging URL
5. Added temporary caching of DEM files from OpenTopo.
6. Began using basic functions in the gri-pipeline gri_shared_library
7. Tweaked the toml and setup_uv.sh files based on current understanding of what is needed for both linux and macOS

## 2026/01/30
1. Resolved relative paths to absolute paths

## 2026/01/21
1. modified run_decision_tree.py to include two arguments on the main function, thereby allowing both manual and programmatic execution.
2. added testing for functionality in run_decision_tree.py
3. three tests are currently commented out while we await synthetic data for imaginary projects
4. resolves a number of bugs which may have emerged due to use of newer versions of dependent packages.
5. Updated code per discussion with Jessica for PR