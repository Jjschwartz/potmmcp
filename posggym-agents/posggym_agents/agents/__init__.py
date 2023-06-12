from posggym_agents.agents.registration import register
from posggym_agents.agents.registration import register_spec
from posggym_agents.agents.registration import make  # noqa
from posggym_agents.agents.registration import spec  # noqa
from posggym_agents.agents.registration import registry  # noqa

from posggym_agents.agents.random import RandomPolicy
from posggym_agents.agents import driving14x14wideroundabout_n2_v0
from posggym_agents.agents import pursuitevasion16x16_v0
from posggym_agents.agents import predatorprey10x10_P2_p3_s2_coop_v0
from posggym_agents.agents import predatorprey10x10_P4_p3_s3_coop_v0


# Generic Random Policies
# ------------------------------
# We don't add the FixedDistributionPolicy since it requires a known
# action distribution which will always be specific to the environment

register(
    id="random-v0",
    entry_point=RandomPolicy,
)


# Driving Policies
# ----------------
for policy_spec in driving14x14wideroundabout_n2_v0.POLICY_SPECS.values():
    register_spec(policy_spec)


# Pursuit Evasion
# ---------------
for policy_spec in pursuitevasion16x16_v0.POLICY_SPECS.values():
    register_spec(policy_spec)


# PredatorPrey
# ------------
for policy_spec in predatorprey10x10_P2_p3_s2_coop_v0.POLICY_SPECS.values():
    register_spec(policy_spec)

for policy_spec in predatorprey10x10_P4_p3_s3_coop_v0.POLICY_SPECS.values():
    register_spec(policy_spec)
