"""Script for running rendering episodes of policies.

The script takes an environment ID and a list of policy ids as arguments.
It then runs and renders episodes.

"""
import argparse
from pprint import pprint

import posggym

import posggym_agents
import posggym_agents.exp as exp_lib


def main(args):    # noqa
    print("\n== Rendering Episodes ==")
    pprint(vars(args))

    env = posggym.make(args.env_id, **{"seed": args.seed})
    policies = []
    for i, policy_id in enumerate(args.policy_ids):
        pi = posggym_agents.make(policy_id, env.model, i)
        policies.append(pi)

    exp_lib.runner.run_episode(
        env,
        policies,
        args.num_episodes,
        trackers=exp_lib.stats.get_default_trackers(),
        renderers=[exp_lib.render.EpisodeRenderer()],
        time_limit=None,
        logger=None,
        writer=None
    )

    print("== All done ==")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id", type=str,
        help="ID of the environment to run experiment in."
    )
    parser.add_argument(
        "-pids", "--policy_ids", type=str, nargs="+",
        help="List of IDs of policies to compare, one for each agent."
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000,
        help="Number of episodes per experiment."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Environment seed."
    )
    main(parser.parse_args())
