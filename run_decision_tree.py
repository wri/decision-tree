import yaml
import json
import pandas as pd

from src.api_utils import get_ids, opentopo_pull_wrapper, get_tm_feats
import src.process_api_results as clean
from src.image_availability import analyze_image_availability
from src.canopy_cover import apply_canopy_classification
from src.slope import apply_slope_classification
import src.decision_trees as tree
import src.cost_calculator as price
import src.weighted_scoring as scoring
import src.update_asana as asana
from src.tools import convert_to_os_path


class VerificationDecisionTree:
    def __init__(self, params_path="params.yaml", secrets_path="secrets.yaml"):
        self.params = self._load_yaml(params_path)
        self.secrets = self._load_yaml(secrets_path)
        self.mode = self.params.get("mode", "full")
        self._resolve_paths()

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_paths(self):
        out = self.params["outfile"]
        cohort = out["cohort"]
        data_v = out["data_version"]
        experiment_id = out["experiment_id"]

        self.portfolio = convert_to_os_path(out["portfolio"].format(cohort=cohort, data_version=data_v))
        self.tm_outfile = convert_to_os_path(out["tm_response"].format(cohort=cohort, data_version=data_v))
        self.slope_stats = convert_to_os_path(out["slope_stats"].format(cohort=cohort,data_version=data_v))
        self.project_feats = convert_to_os_path(out["feats"].format(cohort=cohort, data_version=data_v))
        self.project_feats_maxar = convert_to_os_path(out["feats_maxar"].format(cohort=cohort, data_version=data_v))
        self.maxar_meta = convert_to_os_path(out["maxar_meta"].format(cohort=cohort, data_version=data_v))
        self.tree_results = convert_to_os_path(out["tree_results"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))
        self.poly_score = convert_to_os_path(out["poly_decision"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))
        self.prj_score = convert_to_os_path(out["prj_decision"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))


    def run_decision_tree(self, project_ids):
        if self.mode not in ["full", "partial", "score"]:
            raise ValueError("Invalid mode")

        if self.mode == "full":
            print("Running in FULL mode — acquiring prj data from APIs.")
            input_mode_file = None
        else:
            input_mode_file = self.params['infile']['mode_file']
            if self.mode == "partial":
                print("Running in PARTIAL mode — using cached prj feats.")
            else:
                print("Running in SCORE mode — using cached tree results.")

        if self.mode in ["full", "partial"]:
            if self.mode == "full":
                # uncomment below for testing
                # ids = get_ids(self.params)
                # pd.Series(ids, name="project_id").to_csv(self.portfolio, index=False)

                tm_response = get_tm_feats(self.params, project_ids)
                # uncomment below for testing
                # with open(self.tm_outfile, "w") as f:
                #     json.dump(tm_response, f, indent=4)
                # with open(self.tm_outfile, "r") as f:
                #     tm_response = json.load(f)

                # Clean TM data
                tm_clean = clean.process_tm_api_results(self.params, tm_response)
                # uncomment below for testing
                # tm_clean.to_csv(self.project_feats, index=False)
                # tm_clean.to_csv(self.project_feats_maxar, index=False)
            else:
                tm_clean = pd.read_csv(input_mode_file)

            # compute slope statistics
            slope_statistics = opentopo_pull_wrapper(self.params, self.secrets, tm_clean, process_in_utm_coordinates=True)
            # uncomment below for testing
            # slope_statistics.to_csv(self.slope_stats, index=False)

            # pipeline pause here to get maxar metadata
            ev = compute_ev_statistics(self.params, tm_clean, self.maxar_meta, slope_statistics)
            # uncomment below for testing
            # ev.to_csv(self.tree_results, index=False)

        else:
            slope_statistics = None
            ev = pd.read_csv(input_mode_file)

        # Get results
        scored = scoring.apply_scoring(self.params, ev)
        poly_results = price.calc_cost_to_verify(self.params, scored)
        # uncomment below for testing
        # poly_results.to_csv(self.poly_score, index=False)

        # calculate final project scale decision
        prj_results = scoring.aggregate_project_score(self.params, scored)
        # uncomment below for testing
        # prj_results.to_csv(self.prj_score, index=False)

        # uploads
        if self.params['asana']['upload']:
            asana.update_asana_status_by_gid(self.params, self.secrets, self.prj_score)
        # if self.params['s3']['upload']:
        #     upload_to_s3(self.final_outfile, self.params, self.secrets)  # TODO Jessica - where is this method signature defined in the codebase?
        
        return slope_statistics, poly_results, prj_results


def compute_ev_statistics(params, tm_clean, maxar_meta, slope_statistics):
    branch_images = analyze_image_availability(params, tm_clean, maxar_meta)
    branch_canopy = apply_canopy_classification(params, branch_images)
    branch_slope = apply_slope_classification(params, branch_canopy, slope_statistics)
    baseline = tree.apply_rules_baseline(params, branch_slope)
    ev = tree.apply_rules_ev(params, baseline)

    return ev

def compute_project_results(params, ev):
    scored = scoring.apply_scoring(params, ev)
    poly_results = price.calc_cost_to_verify(params, scored)

    # calculate final project scale decision
    prj_results = scoring.aggregate_project_score(params, scored)

    return poly_results, prj_results


def main(params_file_path: str, secrets_file_path: str = None, parse_only: bool = False):
    workflow = VerificationDecisionTree(params_file_path, secrets_file_path)
    if parse_only:
        return workflow
    else:
        print("Acquiring prj data from APIs.")
        project_ids = get_ids(workflow.params)

        workflow.run_decision_tree(project_ids)
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Main entry point for decision tree.')
    parser.add_argument('--params_yaml_path', metavar='path', required=True,
                        help='Path to params configuration yaml file')
    parser.add_argument('--secrets_yaml_path', metavar='path', required=True,
                        help='Path to secrets configuration yaml file')
    args = parser.parse_args()

    main(args.params_yaml_path, args.secrets_yaml_path)
