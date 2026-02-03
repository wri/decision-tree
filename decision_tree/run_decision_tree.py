import yaml
import pandas as pd


from decision_tree.api_utils import opentopo_pull_wrapper, get_tm_feats, get_ids
import decision_tree.process_api_results as clean
from decision_tree.image_availability import analyze_image_availability
from decision_tree.canopy_cover import apply_canopy_classification
from decision_tree.slope import apply_slope_classification
import decision_tree.decision_trees as tree
import decision_tree.cost_calculator as price
# from src import decision_tree as scoring, decision_tree as asana
import decision_tree.weighted_scoring as scoring
import decision_tree.update_asana as update_asana
from decision_tree.tools import convert_to_os_path


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
        outfile = self.params['outfile']
        cohort = outfile["cohort"]
        data_v = outfile["data_version"]
        experiment_id = outfile["experiment_id"]
        project_data_dir = outfile["project_data_folder"]

        # decision-tree input files
        self.portfolio = convert_to_os_path(project_data_dir, 'portfolio_csvs', outfile["portfolio"].format(cohort=cohort, data_version=data_v))
        self.full_portfolio = convert_to_os_path(project_data_dir, 'portfolio_csvs', f'prj_ids_full_set_{data_v}.csv')
        self.tm_outfile = convert_to_os_path(project_data_dir, 'tm_api_response', outfile['tm_response'].format(cohort=outfile['cohort'], data_version=data_v))
        self.geojson_dir = convert_to_os_path(project_data_dir, None, outfile['geojsons'])
        self.project_feats = convert_to_os_path(project_data_dir, 'feats', outfile["feats"].format(cohort=cohort, data_version=data_v))
        self.maxar_meta = convert_to_os_path(project_data_dir, 'imagery_availability', outfile["maxar_meta"].format(cohort=cohort, data_version=data_v))

        # decision-tree intermediate files
        self.project_feats_maxar = convert_to_os_path(project_data_dir, 'maxar-tools', outfile["feats_maxar"].format(cohort=cohort, data_version=data_v))
        self.slope_stats = convert_to_os_path(project_data_dir, 'slope', outfile["slope_stats"].format(cohort=cohort, data_version=data_v))
        self.tree_results = convert_to_os_path(project_data_dir, 'tree_output', outfile["tree_results"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))
        self.poly_score = convert_to_os_path(project_data_dir, 'decision_scores', outfile["poly_decision"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))

        # decision-tree output files
        self.prj_score = convert_to_os_path(project_data_dir, 'decision_scores', outfile["prj_decision"].format(cohort=cohort, data_version=data_v, experiment_id=experiment_id))

        # rules
        self.rules_file_path = convert_to_os_path(project_data_dir, None, self.params['criteria']['rules'])

    def run_decision_tree(self, project_ids):
        if self.mode not in ["full", "partial", "score"]:
            raise ValueError("Invalid mode")

        if self.mode == "full":
            print("Running in FULL mode — acquiring prj data from APIs.")
            input_mode_file = None
        else:
            if self.mode == "partial":
                print("Running in PARTIAL mode — using cached prj feats.")
                input_mode_file = self.project_feats
            else:
                print("Running in SCORE mode — using cached tree results.")
                input_mode_file = self.tree_results

        if self.mode in ["full", "partial"]:
            if self.mode == "full":
                project_ids = get_ids(self.params)
                # uncomment below for testing
                # project_ids = project_ids[:n] # if desired, specify the number of projects to test
                # pd.Series(project_ids, name="project_id").to_csv(self.full_portfolio, index=False)

                tm_response = get_tm_feats(self.params, self.geojson_dir, self.tm_outfile, project_ids)
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
            slope_statistics = opentopo_pull_wrapper(self.params, self.secrets, self.geojson_dir, tm_clean, process_in_utm_coordinates=True)
            # uncomment below for testing
            # slope_statistics.to_csv(self.slope_stats, index=False)

            # pipeline pause here to get maxar metadata
            ev = compute_ev_statistics(self.params, self.rules_file_path, tm_clean, self.maxar_meta, slope_statistics)
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
            update_asana.update_asana_status_by_gid(self.params, self.secrets, self.prj_score)
        # if self.params['s3']['upload']:
        #     upload_to_s3(self.final_outfile, self.params, self.secrets)  # TODO Jessica - where is this method signature defined in the codebase?

        return slope_statistics, poly_results, prj_results


def compute_ev_statistics(params, rules_file_path, tm_clean, maxar_meta, slope_statistics):
    branch_images = analyze_image_availability(params, tm_clean, maxar_meta)
    branch_canopy = apply_canopy_classification(params, branch_images)
    branch_slope = apply_slope_classification(params, branch_canopy, slope_statistics)
    baseline = tree.apply_rules_baseline(rules_file_path, branch_slope)
    ev = tree.apply_rules_ev(params, rules_file_path, baseline)

    return ev

def compute_project_results(params, ev):
    scored = scoring.apply_scoring(params, ev)
    poly_results = price.calc_cost_to_verify(params, scored)

    # calculate final project scale decision
    prj_results = scoring.aggregate_project_score(params, scored)

    return poly_results, prj_results


