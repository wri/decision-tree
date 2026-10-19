from decision_tree.run_decision_tree import STEPS, VerificationDecisionTree, main as run_main


def main(params_file_path: str,
         secrets_file_path: str | None = None,
         parse_only: bool = False,
         step: str = "all",
         project_ids: list[str] | None = None) -> VerificationDecisionTree | None:
    return run_main(params_file_path, secrets_file_path, parse_only=parse_only,
                    step=step, project_ids=project_ids)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Main entry point for decision tree.')
    parser.add_argument('--params_yaml_path', metavar='path', required=True,
                        help='Path to params configuration yaml file')
    parser.add_argument('--secrets_yaml_path', metavar='path', required=True,
                        help='Path to secrets configuration yaml file')
    parser.add_argument('--step', choices=STEPS, default='all',
                        help="'polygon' = polygon-level metrics (automated trigger); "
                             "'project' = project-level metrics from existing polygon output (operator trigger); "
                             "'all' = both steps (default)")
    parser.add_argument('--project_ids', nargs='+', default=None,
                        help="Project ids. Required by the polygon step in 'projectids' mode "
                             "and by the project step (projects to aggregate)")
    args = parser.parse_args()

    main(args.params_yaml_path, args.secrets_yaml_path,
         step=args.step, project_ids=args.project_ids)
