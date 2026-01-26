from decision_tree_src.api_utils import get_ids
from decision_tree_src.run_decision_tree import VerificationDecisionTree


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