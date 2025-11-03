# TerraFund Decision Tree

## Repository Overview

This repository contains the code for running a rule-based decision process to identify the appropriate verification method (field or remote) for a TerraFund project.

## 🧠 Purpose & Scope

The primary functions of this code are located within `run_decision_tree.py`. 
Data Gathering & Cleaning
  - Query the TerraMatch API, Maxar API and OpenTopo API to gather input data for the decision tree
  - Process, validate and clean the API response into the various "branches" of the tree
Apply logic
  - Apply the rules framework
Create decisions
  - Apply the weighted scoring approach at the polygon level
  - Calculate the cost to monitor the project
  - Aggregate polygon scores to derive project score
Upload
  - Push results to Asana
  - Push results to s3


## 🎯 Goals & Non-Goals

* **Goals**: create an individual verification score for each project in the TerraFund portfolio.
* **Non-Goals:** score projects that are not part of TerraFund. 


## 📦 Inputs & Upstreams

### Upstream Repositories

| Repo | Artifact Provided | Interface | Versioning/Contract | Notes |
|----------------------------|-------------------|-----------|---------------------|-------|
| maxar-tools| API response| --- | ------------------- | ----- |
| terramatch-researcher-api | API response| -- | ------------------- | ----- |


### External Data Sources

| Source | Type (API/S3/etc.) | Endpoint/Path | Auth | Data Format | Schema Link |
|--------|-------------------|-----------------|-----|-------| ----|
| OpenTopography | API | ------------- | API key | geotiff | ----------- |

### Expected Input Formats

* mode dependent - updated params.yaml file or project id csv for processing
* rule_template.csv

### Input Validation

* Schema checks: `process_api_results.py` checks dtypes, missing or incomplete data and duplicates
* Range checks: None
* Null/duplicate handling: `process_api_results.py` checks dtypes, missing or incomplete data and duplicates
* Failure behavior (fail fast vs warn): None
* **Upstream repos:**: `tf-biophysical-indicators`, `terramatch-researcher-api`, `maxar-tools`
* **External data sources:** (S3/API/DB): `Opentopo API`
* **Expected input format(s):** (e.g., Parquet, GeoTIFF, JSON) 
* **Input validation logic:** (schema checks, nulls, ranges):

## 📤 Outputs & Downstreams

### Produced Artifacts

| Artifact                 | Format | Path/Location                                                        | Update Frequency | Size     | Retention |
|--------------------------|--------|-----------------------------------------------------------------------|------------------|----------|-----------|
| project ids     | csv| `data/portfolio_csvs/prj_ids_{cohort}_{data_version}.csv`            | N/A              | x   | Permanent |
| tm API response   | json    | `data/tm_api_response/{cohort}_{data_version}.json`         | N/A              | x   | Permanent |
| tn clean response | csv    | `data/feats/tm_api_{cohort}_{data_version}.csv`                | N/A              | x  | Permanent |
| project polygon | geojson  | `s3://tree_verification/output/project_data/SHORTNAME/geojson/SHORTNAME_{date}.geojson` | N/A | x | Permanent |
| img metadata  | csv    | `data/imagery_availability/comb_img_availability_{cohort}_{data_version}.csv`   | N/A | x    | Permanent |
| slope statistics   | csv    | `data/slope/slope_statistics_{cohort}_{data_version}.csv`   | N/A   | x   | Permanent |
| decision scores  | csv    | `data/decision_scores/prj_output_{cohort}_{data_version}_{experiment_id}.csv`  | N/A   | x   | Permanent |


### Downstream Repos / Consumers

| Repo | Consumed Artifact | Contract Type | Dependency Level (strong/soft) |
| tree-verification | binary decision | ---- | strong |
| maxar-tools (image ordering) | binary decision | ---- | strong |
| maxar-tools (image availability) | image count | ---- | soft |


### Output Guarantees / Completeness checks

* Schema stability: None
* Freshness SLO: none
* Completeness / QA checks before publish: none
* **Artifacts produced:** (files, tables, models):

## ⚙️ Configuration & Secrets

### Config Files
**Config files:** `params.yaml`, `rule_template.csv`
```yaml
criteria:
  canopy_threshold: 60                                        
  cloud_thresh: 50                                   
  off_nadir: 30                                               
  sun_elevation: 30
  img_count: 1                                                 
  baseline_range: [-365, 0]                                
  ev_range: [730, 1095]                                        
  drop_missing: False
  slope_thresh: 20
  rules: data/rule_template.csv

```

### Configuration Keys

Don't actually list the config keys in plain text but describe what is needed. 

| Key | Type | Default | Required | Description |
| --- | ---- | ------- | :------: | ----------- |

### Secrets Handling
**Config files:** `secrets.yaml` 

```yaml
access_token:
opentopo_key: 
aws:
  aws_access_key_id: 
  aws_secret_access_key:
  aws_region:
asana:
  pat: 
```


## 🚀 Execution

### Entry Points

| Command | Purpose |
|----------|--------|
| python3 run_decision_tree.py | ------- |


### Orchestration

There is currently no orchestration (manual trigger via CLI) but this pipeline needs to be run after the polygons have been approved on TerraMatch and the tree cover indicator has been calculated.

| System | Trigger | Frequency | Dependency Triggered? |
| ------ | ------- |-----------|-----|
| N/A | CLI | Manual | No |

### Runtime & Cost Expectations

* Expected runtime: tbd
* GPU/CPU requirements: 
* Storage footprint: ~1GB
* Cost control levers (batch size, concurrency): N/A
* **Entry point(s):** 
* **CLI usage:** (example commands)
* **Schedulers / Orchestrators:** (Airflow / Cron / GitHub Actions): 
* **Runtime + cost notes:** 

## 🏗️ Environment & Dependencies

### Python & System Environment

can create a requirements.txt when the time comes.

## 🔍 Observability

### Logging

The logging for this repository should ideally create a .jsonl file within each `SHORTNAME` directory specifying success / failure.

* Format: Currently only logging via print statements and capture to a txt file via `python3 main.py > tmp_log.txt`
* Logging levels: Singular
* Sample log message:

### Metrics

| Metric | Type | Labels | Description |
| ------ | ---- | ------ | ----------- |

### Dashboards/Monitoring

* Links
* Alerting rules
* **Logging:** (format, sinks)

## 💥 Failure Modes & Recovery

### Expected Failures & Behavior

Failure Case
*

### Checkpoints & Resume

* Where checkpoints stored: 
* Resume instructions:
* Common failure scenarios

## 👥 Ownership & Access

### Owners

* Engineering owner: Jessica Ertel
* Data owner
* On-call/rotation (if exists)

### Access Model

* Code access
* Data bucket access
* Secrets access


## 🗺️ Pipeline Position

### Data Flow

```
<Upstream repo/data> → This repo → <Downstream repo/data>
```

* Sync frequency with upstream/downstream repos
* Dependency risks