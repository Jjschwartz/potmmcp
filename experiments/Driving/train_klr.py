import baposgmcp.rllib as ba_rllib

from rl_config import get_rl_training_config


if __name__ == "__main__":
    parser = ba_rllib.get_train_klr_exp_parser()
    args = parser.parse_args()

    ba_rllib.train_klr_policy(
        args.env_name,
        k=args.k,
        best_response=args.train_best_response,
        is_symmetric=True,
        seed=args.seed,
        trainer_config=get_rl_training_config(
            args.env_name, args.seed, args.log_level
        ),
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        num_iterations=args.num_iterations,
        run_serially=args.run_serially,
        save_policies=args.save_policies,
        verbose=True
    )
