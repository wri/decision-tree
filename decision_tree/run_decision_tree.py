import os

import yaml
import pandas as pd
import json
from pathlib import Path

from gri_shared_library.os_tools import create_folder

from decision_tree.api_utils import opentopo_pull_wrapper, get_geoparquet
import decision_tree.process_api_results as clean
from decision_tree.image_availability import analyze_image_availability 
from decision_tree.canopy_cover import apply_canopy_classification
from decision_tree.slope import apply_slope_classification
from decision_tree.s3_utils import upload_to_s3
import decision_tree.polygon_decisions as poly_tree
import decision_tree.cost_calculator as price
import decision_tree.project_decisions as proj_tree
import decision_tree.update_asana as update_asana
from decision_tree.tools import convert_to_os_path, load_secrets

class Checkpointer:
    """
    Controls optional persistence of intermediate files.

    When enabled, save() writes a keyed DataFrame to its resolved path and
    load() reads it back. When disabled, save() is a no-op and the pipeline
    runs end-to-end in memory.

    Keys map to the intermediate paths defined in VerificationDecisionTree._checkpoint_paths().
    """

    def __init__(self, enabled: bool, paths: dict):
        self.enabled = enabled
        self.paths = paths

    def save(self, key: str, df: pd.DataFrame, always: bool = False):
        if self.enabled or always:
            path = self.paths[key]

            # Create target folder
            target_folder = os.path.dirname(path)
            create_folder(target_folder)

            df.to_csv(path, index=False)
            print(f"[checkpoint] {key} → {path}")

    def load(self, key: str) -> pd.DataFrame:
        path = self.paths[key]
        if not Path(path).exists():
            raise FileNotFoundError(f"[checkpoint] No file found for '{key}' at {path}")
        print(f"[checkpoint] loading {key} from {path}")
        return pd.read_csv(path)

    def exists(self, key: str) -> bool:
        return key in self.paths and Path(self.paths[key]).exists()


class VerificationDecisionTree:
    def __init__(self, params_path="params.yaml", secrets_path="secrets.yaml", checkpoint=False):
        self.params = self._load_yaml(params_path)
        self.secrets = load_secrets(secrets_path)
        self.mode = self.params.get("mode", "full")
        self._resolve_paths()
        self.checkpoint = Checkpointer(enabled=checkpoint, paths=self._checkpoint_paths())

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_paths(self):
        outfile = self.params['outfile']
        cohort = outfile["cohort"]
        data_v = outfile["data_version"]
        experiment_id = outfile["experiment_id"]
        project_data_dir = outfile["project_data_folder"]

        # decision-tree input files for full or projectids mode
        self.portfolio = convert_to_os_path(project_data_dir, outfile["portfolio"].format(cohort=cohort, data_version=data_v))
        self.tm_raw = convert_to_os_path(project_data_dir, outfile['geoparquet'].format(data_version=data_v))
        self.maxar_meta = convert_to_os_path(project_data_dir, outfile["maxar_meta"].format(cohort=cohort, data_version=data_v))

        # decision-tree intermediate files
        self.geojson_dir = convert_to_os_path(project_data_dir, outfile['geojsons'])
        self.feats = convert_to_os_path(project_data_dir, outfile["feats"].format(cohort=cohort, data_version=data_v))
        self.project_feats_maxar = convert_to_os_path(project_data_dir, outfile["feats_maxar"].format(cohort=cohort, data_version=data_v))
        self.slope_stats = convert_to_os_path(project_data_dir, outfile["slope_stats"].format(cohort=cohort, data_version=data_v))
        self.tree_results = convert_to_os_path(project_data_dir, outfile["tree_results"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))
        
        # output files
        self.poly_score = convert_to_os_path(project_data_dir, outfile["poly_decision"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))
        self.prj_score = convert_to_os_path(project_data_dir, outfile["prj_decision"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))

        # rules
        self.rules = convert_to_os_path(project_data_dir, self.params['criteria']['rules'])

    def _checkpoint_paths(self) -> dict:
        """
        Maps checkpoint keys to their resolved file paths.
        Intermediates (feats, slope_stats, tree_results) are written only when checkpoint=True.
        Final outputs (poly_score, prj_score) are always written via always=True in save().
        """
        return {
            # intermediates
            "feats":        self.feats,
            "slope_stats":  self.slope_stats,
            "tree_results": self.tree_results,
            # final outputs
            "poly_score":   self.poly_score,
            "prj_score":    self.prj_score,
        }

    def run_decision_tree(self, project_ids: list[str] = None, limit_to_test_projects: bool = False):
        if self.mode not in ["full", "score", "projectids"]:
            raise ValueError("Invalid mode")

        if self.mode in ["full", "score"] and not (project_ids is None or project_ids == []):
            raise ValueError(f"The project_id parameter cannot be specified for the '{self.mode}' mode.")

        if self.mode == "projectids" and (project_ids is None or project_ids == []):
            raise ValueError("The project_id parameter must be specified for 'projectids' mode")

        slope_statistics = None
        if self.mode in ("full", "projectids"):
            print(f"Running in {self.mode.upper()} mode — acquiring prj data.")
            tm_raw_path = get_geoparquet(self.params, self.secrets, self.tm_raw)
            tm_clean = clean.process_tm_results(self.params, 
                                                tm_raw_path, 
                                                self.geojson_dir, 
                                                project_ids, 
                                                limit_to_test_projects)

            self.checkpoint.save("feats", tm_clean)
            slope_statistics = opentopo_pull_wrapper(self.params, 
                                                     self.secrets, 
                                                     self.geojson_dir, 
                                                     tm_clean, 
                                                     process_in_utm_coordinates=True)
            self.checkpoint.save("slope_stats", slope_statistics)

            # pipeline pause here to get maxar metadata
            ev = compute_branches(self.params, 
                                  self.rules, 
                                  tm_clean, 
                                  self.maxar_meta, 
                                  slope_statistics)
            self.checkpoint.save("tree_results", ev)

        elif self.mode == "score":
            print("Running in SCORE mode — using cached prj data.")
            ev = pd.read_csv(self.tree_results)

        # Get results
        poly_results, prj_results = compute_project_results(self.params, ev)
        self.checkpoint.save("poly_score", poly_results, always=True)
        self.checkpoint.save("prj_score", prj_results, always=True)

        # uploads
        if self.params['asana']['upload']:
            update_asana.update_asana_status_by_gid(self.params, self.secrets, self.prj_score)
        if self.params['s3']['upload']:
            upload_to_s3(self.prj_score, self.params, self.secrets) 

        return poly_results, prj_results 


def compute_branches(params, rules_file_path, tm_clean, maxar_meta, slope_statistics):
    """Run decision tree branch logic."""
    branch_images = analyze_image_availability(params, tm_clean, maxar_meta)
    branch_canopy = apply_canopy_classification(params, branch_images)
    branch_slope = apply_slope_classification(params, branch_canopy, slope_statistics)
    baseline = poly_tree.apply_rules_baseline(rules_file_path, branch_slope)
    ev = poly_tree.apply_rules_ev(params, rules_file_path, baseline)
    return ev

def compute_project_results(params, ev):
    """Run decision scoring."""
    scored = poly_tree.apply_scoring(params, ev)
    poly_results = price.calc_cost_to_verify(params, scored)
    prj_results = proj_tree.aggregate_project_score(params, scored)
    return poly_results, prj_results

def main(params_file_path: str, secrets_file_path: str = None, parse_only: bool = False):
    workflow = VerificationDecisionTree(params_file_path, secrets_file_path)
    if parse_only:
        return workflow
    else:
        workflow.run_decision_tree(None)
        return None

