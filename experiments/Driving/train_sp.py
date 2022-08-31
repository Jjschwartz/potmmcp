import baposgmcp.rllib as ba_rllib

from rl_config import get_rl_training_config


if __name__ == "__main__":
    parser = ba_rllib.get_train_sp_exp_parser()
    args = parser.parse_args()

    ba_rllib.train_sp_policy(
        args.env_name,
        seed=args.seed,
        trainer_config=get_rl_training_config(
            args.env_name, args.seed, args.log_level
        ),
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        num_iterations=args.num_iterations,
        save_policy=args.save_policy,
        verbose=True
    )
