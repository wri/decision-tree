import yaml
import json
import pandas as pd

from src.api_utils import get_ids, tm_pull_wrapper, opentopo_pull_wrapper
import src.process_api_results as clean
from src.image_availability import analyze_image_availability
from src.canopy_cover import apply_canopy_classification
from src.slope import apply_slope_classification
import src.decision_trees as tree
import src.cost_calculator as price
import src.weighted_scoring as scoring
import src.update_asana as asana

class VerificationDecisionTree:
    def __init__(self, params_path="params.yaml", secrets_path="secrets.yaml"):
        self.params = self._load_yaml(params_path)
        self.secrets = self._load_yaml(secrets_path)
        self._resolve_paths()

    def _load_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_paths(self):
        out = self.params["outfile"]
        cohort = out["cohort"]
        data_v = out["data_version"]
        experiment_id = out["experiment_id"]

        self.portfolio = out["portfolio"].format(cohort=cohort, data_version=data_v)
        self.tm_outfile = out["tm_response"].format(cohort=cohort, data_version=data_v)
        self.slope_stats = out["slope_stats"].format(cohort=cohort,data_version=data_v)
        self.project_feats = out["feats"].format(cohort=cohort, data_version=data_v)
        self.project_feats_maxar = out["feats_maxar"].format(cohort=cohort, data_version=data_v)
        self.maxar_meta = out["maxar_meta"].format(cohort=cohort, data_version=data_v)
        self.tree_results = out["tree_results"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id)
        self.poly_score = out["poly_decision"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id)
        self.prj_score = out["prj_decision"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id)

    def run(self, save_to_asana: bool):
        print("Acquiring prj data from APIs.")
        ids = get_ids(self.params)
        pd.Series(ids, name="project_id").to_csv(self.portfolio, index=False)
        #ids = pd.read_csv("data/portfolio_csvs/prj_ids_c1_06-30-2025.csv")
        ids = list(set(ids.project_id))

        tm_response = tm_pull_wrapper(self.params, ids)
        with open(self.tm_outfile, "w") as f:
            json.dump(tm_response, f, indent=4)

        with open(self.tm_outfile, "r") as f:
            tm_response = json.load(f)

        # Clean TM data and save to csv files
        tm_clean = clean.process_tm_api_results(self.params, tm_response)
        tm_clean.to_csv(self.project_feats, index=False)
        tm_clean.to_csv(self.project_feats_maxar, index=False)

        # compute slope statistics
        slope_statistics = opentopo_pull_wrapper(self.params, self.secrets, tm_clean)
        slope_statistics.to_csv(self.slope_stats, index=False)

        # pipeline pause here to get maxar metadata

        # compute ev decision
        ev = compute_ev_statistics(self.params, tm_clean, self.maxar_meta, slope_statistics)
        ev.to_csv(self.tree_results, index=False)

        # Get results and save to csv files
        poly_results, prj_results = compute_project_results(self.params, ev)
        poly_results.to_csv(self.poly_score, index=False)
        prj_results.to_csv(self.prj_score, index=False) 

        # uploads
        if save_to_asana:
            asana.update_asana_status_by_gid(self.params, self.secrets, self.prj_score)
        # upload_to_s3.upload_results_to_s3(self.final_outfile, self.params, self.secrets)

def compute_ev_statistics(params, tm_clean, maxar_meta, slope_statistics):
    branch_images = analyze_image_availability(params, tm_clean, maxar_meta)  # combine feats + imgs
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


def main(params_file_path: str, secrets_file_path: str = None, parse_only: bool = False, save_to_asana: bool = True):
    workflow = VerificationDecisionTree(params_file_path, secrets_file_path)
    if parse_only:
        return workflow
    else:
        workflow.run(save_to_asana)
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
