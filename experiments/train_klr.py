import importlib

import baposgmcp.rllib as ba_rllib


if __name__ == "__main__":
    parser = ba_rllib.get_train_klr_exp_parser()
    parser.add_argument(
        "--config", type=str,
        help=(
            "Module path to python file containing training config "
            "(e.g. Driving7x7RoundAbout-n2-v0.rl_config)."
        )
    )
    args = parser.parse_args()

    config_lib = importlib.import_module(args.config)
    assert hasattr(config_lib, "get_rl_training_config"), \
        "Config file must define function 'get_rl_training_config'."

    ba_rllib.train_klr_policy(
        args.env_name,
        k=args.k,
        best_response=args.train_best_response,
        is_symmetric=True,
        seed=args.seed,
        trainer_config=config_lib.get_rl_training_config(
            args.env_name, args.seed, args.log_level
        ),
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        num_iterations=args.num_iterations,
        run_serially=args.run_serially,
        save_policies=args.save_policies,
        verbose=True
    )
