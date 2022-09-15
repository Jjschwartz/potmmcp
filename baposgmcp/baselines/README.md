# Baselines

## Fixed Policies

This baseline is the performance of the individual policies that make up the
meta-policy against each coplayer policy. This provides a baseline for each
component policy that BAPOSGMCP uses.

Depending on the individual policies, they may provide upper bound and lowe
bounds on performance against other individual policies. For example, a policy
A that is a best response against policy B will provide an upper bound for
performance against policy B, in addition in competitive environments policy B's
performance may be a lower bound.

We also look at the expected performance of each fixed policy against the prior
over policies. This gives a baseline of how well any individual policy does
given the other agent's policy is unknown and only the prior is known.


## Prior + Meta Policy

This baseline selects the fixed policy according to the Policy-Prior and
Meta-Policy. Specifically, for each episode a policy state is sampled from the
prior and then the fixed policy to use for the episode is sampled from the
Meta-Policy.

This provides baseline performance of BAPOSGMCP without using online planning
(beliefs + search).


## PO-Meta

This baseline maintains a belief over the history-policy-state and uses this
to compute the policy for the next step by taking an expectation over the
Meta-Policy w.r.t the belief over policy-states.

This provides baseline performance of BAPOSGMCP using beliefs and the
meta-policy but without search.


## PO-MetaRollout

This baseline maintains a belief over the history-policy-state similar to
PO-Meta and BAPOSGMCP. Each step it runs N rollouts from the root node using
the meta-policy for rollouts and different methods for selecting the actions
to explore from the root node.

This provides baseline performance of BAPOSGMCP using beliefs and the
meta-policy and one-step lookahead search.

*Action Selection*

1. PUCB
2. UCB
3. Uniform


## BAPOSGMCP - UCB

This baseline is the same as POSGMCP except uses UCB action selection instead
of PUCB.

This provides comparison for seeing the benefit of PUCB over UCB action
selection in this problem.
