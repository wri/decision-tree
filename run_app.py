from decision_tree.run_decision_tree import main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Main entry point for decision tree.')
    parser.add_argument('--params_yaml_path', metavar='path', required=True,
                        help='Path to params configuration yaml file')
    parser.add_argument('--secrets_yaml_path', metavar='path', required=True,
                        help='Path to secrets configuration yaml file')
    args = parser.parse_args()

    main(args.params_yaml_path, args.secrets_yaml_path)
