import importlib

from posggym_agents.rllib import get_train_klr_exp_parser, train_klr_policy


if __name__ == "__main__":
    parser = get_train_klr_exp_parser()
    parser.add_argument(
        "--config", type=str,
        help=(
            "Module path to python file containing training config (e.g. "
            "posggym_agents.agents.driving7x7roundabout_n2_v0.train_config)."
        )
    )
    args = parser.parse_args()

    config_lib = importlib.import_module(args.config)
    assert hasattr(config_lib, "get_rl_training_config"), \
        "Config file must define function 'get_rl_training_config'."

    train_klr_policy(
        args.env_id,
        k=args.k,
        best_response=args.train_best_response,
        seed=args.seed,
        trainer_config=config_lib.get_rl_training_config(
            args.env_id, args.seed, args.log_level
        ),
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        num_iterations=args.num_iterations,
        run_serially=args.run_serially,
        save_policies=args.save_policies,
        verbose=True
    )
