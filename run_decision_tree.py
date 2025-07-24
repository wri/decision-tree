import yaml
import json
import sys
import pandas as pd
sys.path.append('src/')
from api_utils import get_ids
from api_utils import tm_pull_wrapper, opentopo_pull_wrapper
import process_api_results as clean
from image_availability import analyze_image_availability
from canopy_cover import apply_canopy_classification
from slope import apply_slope_classification
import decision_trees as tree

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
        self.portfolio = out["portfolio"].format(cohort=out["cohort"], today=out["today"])
        self.tm_outfile = out["tm_response"].format(cohort=out["cohort"], today=out["today"])
        self.slope_stats = out["slope_stats"].format(cohort=out["cohort"], today=out["today"])
        self.project_feats = out["feats"].format(cohort=out["cohort"], today=out["today"])
        self.project_feats_maxar = out["feats_maxar"].format(cohort=out["cohort"], today=out["today"])
        self.maxar_meta = out["maxar_meta"].format(cohort=out["cohort"], today=out["today"])
        self.final_outfile = out["decision"].format(cohort=out["cohort"], today=out["today"])

    def run(self):
        # ids = get_ids(self.params)
        # pd.Series(ids, name="project_id").to_csv(self.portfolio, index=False)
        # ids = pd.read_csv("data/portfolio_csvs/prj_ids_c1_06-30-2025.csv")
        # ids = list(set(ids.project_id))
        
        # tm_response = tm_pull_wrapper(self.params, ids)
        # with open(self.tm_outfile, "w") as f:
        #     json.dump(tm_response, f, indent=4)
     
        # tm_clean = clean.process_tm_api_results(self.params, tm_response) # cleans TM response
        # tm_clean.to_csv(self.project_feats, index=False)
        # tm_clean.to_csv(self.project_feats_maxar, index=False) 
        tm_clean = pd.read_csv('data/feats/tm_api_c1_07-14-2025.csv')
        print(tm_clean.project_name.unique())

        slope_statistics = opentopo_pull_wrapper(self.params, self.secrets, tm_clean) # downloads and cleans slope
        slope_statistics.to_csv(self.slope_stats, index=False)

        # pipeline pause here to get maxar metadata
        branch_images = analyze_image_availability(self.params, tm_clean, self.maxar_meta) # combine feats + imgs
        branch_canopy = apply_canopy_classification(self.params, branch_images)
        branch_slope = apply_slope_classification(self.params, branch_canopy, slope_statistics)

        results = tree.apply_rules_baseline(self.params, branch_slope)
        results.to_csv(self.final_outfile, index=False)

        # upload to s3 and trigger asana API
        # upload_to_s3.upload_results_to_s3(self.final_outfile, self.params, self.secrets)


def main():
    workflow = VerificationDecisionTree()
    workflow.run()


if __name__ == "__main__":
    main()
