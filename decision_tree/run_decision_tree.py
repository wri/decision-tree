import os
from pathlib import Path
import datetime
import pandas as pd
import yaml
from gri_shared_library.os_tools import create_folder, get_project_ids_from_geoparquet
from tm_api_utils.tm_features import get_tm_feats

import decision_tree.cost_calculator as price
import decision_tree.polygon_decisions as poly_tree
import decision_tree.process_api_results as clean
import decision_tree.project_decisions as proj_tree
import decision_tree.update_asana as update_asana
from decision_tree.api_utils import download_geoparquet
from decision_tree.canopy_cover import apply_canopy_classification
from decision_tree.image_availability import analyze_image_availability
#from decision_tree.slope import  opentopo_pull_wrapper, apply_slope_classification
from decision_tree.slope_new import copernicus_pull_wrapper, apply_slope_classification
from decision_tree.tools import convert_to_os_path, load_secrets, get_tm_auth, load_yaml
from decision_tree.constants import RULES, TestProjectHandling

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
        self.params = load_yaml(params_path)
        self.secrets = load_secrets(secrets_path)
        self.mode = self.params.get("mode", "full")
        self.tm_source = self.params["tm_source"]
        self._resolve_paths()
        self.checkpoint = Checkpointer(enabled=checkpoint, paths=self._checkpoint_paths())

    def _resolve_paths(self):
        outfile = self.params['outfile']
        self.cohort = outfile["cohort"]
        data_v = outfile["data_version"]
        experiment_id = outfile["experiment_id"]
        project_data_dir = outfile["project_data_folder"]

        # decision-tree input files for full or projectids mode
        self.portfolio = convert_to_os_path(project_data_dir, outfile["portfolio"].format(cohort=self.cohort, data_version=data_v))

        if self.tm_source.lower() == "geoparquet":
            tm_data_v = data_v
        else:
            now = datetime.datetime.now()
            tm_data_v = now.strftime("%Y-%m-%d-%H-%M")
        self.tm_raw = convert_to_os_path(project_data_dir, outfile['geoparquet'].format(data_version=tm_data_v))
        self.tm_outfile = convert_to_os_path(project_data_dir, outfile['tm_response'].format(cohort=outfile['cohort'],
                                                                                             data_version=data_v))

        self.maxar_meta = convert_to_os_path(project_data_dir,
                                             outfile["maxar_meta"].format(cohort=self.cohort, data_version=data_v))

        # decision-tree intermediate files
        self.geojson_dir = convert_to_os_path(project_data_dir, outfile['geojsons'])
        self.feats = convert_to_os_path(project_data_dir,
                                        outfile["feats"].format(cohort=self.cohort, data_version=data_v))
        self.project_feats_maxar = convert_to_os_path(project_data_dir,
                                                      outfile["feats_maxar"].format(cohort=self.cohort, data_version=data_v))
        self.slope_stats = convert_to_os_path(project_data_dir,
                                              outfile["slope_stats"].format(cohort=self.cohort, data_version=data_v))
        self.tree_results = convert_to_os_path(project_data_dir,
                                               outfile["tree_results"].format(cohort=self.cohort, data_version=data_v, experiment_id=experiment_id))
        
        # output files
        self.poly_score = convert_to_os_path(project_data_dir,
                                             outfile["poly_decision"].format(cohort=self.cohort, data_version=data_v, experiment_id=experiment_id))
        self.prj_score = convert_to_os_path(project_data_dir,
                                            outfile["prj_decision"].format(cohort=self.cohort, data_version=data_v, experiment_id=experiment_id))

        # rules template
        self.rules = convert_to_os_path("", RULES)

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

    def run_decision_tree(self, project_ids: list[str] = None, test_project_handling = TestProjectHandling.EXCLUDE):
        if self.mode in ["full", "score"] and not (project_ids is None or project_ids == []):
            raise ValueError(f"The project_id parameter cannot be specified for the '{self.mode}' mode.")

        if self.mode == "projectids" and (project_ids is None or project_ids == []):
            raise ValueError("The project_id parameter must be specified for 'projectids' mode")

        slope_statistics = None
        if self.mode in ("full", "projectids"):
            print(f"Running in {self.mode.upper()} mode — acquiring prj data.")
            download_geoparquet(self.params, self.secrets, self.tm_raw)

            if self.tm_source.lower() == 'api':
                expanded_cohort = 'terrafund-cohort-1' if self.cohort == 'c1' else 'terrafund-cohort-2'
                if self.mode == 'full':
                    project_ids = get_project_ids_from_geoparquet(self.tm_raw, expanded_cohort)

                auth_headers = get_tm_auth()
                tm_response = get_tm_feats(auth_headers=auth_headers, project_ids=project_ids)
                tm_response['cohort'] = f'["{expanded_cohort}"]'
            else:
                tm_response = clean._read_geoparquet(self.tm_raw)

            tm_clean = clean.process_tm_results(self.params,
                                                tm_response,
                                                self.geojson_dir,
                                                project_ids,
                                                test_project_handling= test_project_handling)

            self.checkpoint.save("feats", tm_clean)
            # slope_statistics = opentopo_pull_wrapper(self.params, 
            #                                          self.secrets, 
            #                                          self.geojson_dir, 
            #                                          tm_clean, 
            #                                          process_in_utm_coordinates=True)

            slope_statistics = copernicus_pull_wrapper(self.params,
                                                       self.geojson_dir,
                                                       tm_clean,
                                                       )
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
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Get results
        poly_results, prj_results = compute_project_results(self.params, ev)
        self.checkpoint.save("poly_score", poly_results, always=True)
        self.checkpoint.save("prj_score", prj_results, always=True)

        # uploads
        if self.params['asana']['upload']:
            update_asana.update_asana_status_by_gid(self.params, self.secrets, self.prj_score)
        if self.params['s3']['upload']:
            raise Exception("The upload to S3 option is currently not supported.")
            # TODO The function call signature needs to be corrected.
            # upload_to_s3(self.prj_score, self.params, self.secrets)

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
    poly_results = price.calc_cost_to_verify(scored)
    prj_results = proj_tree.aggregate_project_score(params, scored)
    return poly_results, prj_results

def main(params_file_path: str, secrets_file_path: str = None, parse_only: bool = False):
    workflow = VerificationDecisionTree(params_file_path, secrets_file_path)
    if parse_only:
        return workflow
    else:
        workflow.run_decision_tree(None)
        return None

