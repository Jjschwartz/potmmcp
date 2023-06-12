import re
import copy
from typing import Dict, Optional, Protocol, Sequence, Union, List

import posggym.model as M
from posggym import error, logger

from posggym_agents.policy import BasePolicy


# [env-name/](policy-id)-v(version)
# env-name is group 1, policy_id is group 2, version is group 3
POLICY_ID_RE: re.Pattern = re.compile(r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")


class PolicyEntryPoint(Protocol):
    """Entry point function for instantiating a new policy instance."""

    def __call__(self,
                 model: M.POSGModel,
                 agent_id: M.AgentID,
                 policy_id: str,
                 **kwargs) -> BasePolicy: ...


class PolicySpec:
    """A specification for a particular agent policy.

    Used to register agent policies that can then be dynamically loaded.

    Arguments
    ---------
    id (str):
        The ID of the agent policy
    entry_point (PolicyEntryPoint):
        The Python entrypoint for initializing an instance of the agent policy
    valid_agent_ids (Optional[List[M.AgentID]):
        Optional AgentIDs in environment that policy is compatible with. If
        None then assumes policy can be used for any agent in the environment.
    kwargs (Optional[Dict])):
        The kwargs, if any, to pass to the agent initializing

    """

    def __init__(self,
                 id: str,
                 entry_point: PolicyEntryPoint,
                 valid_agent_ids: Optional[
                     Union[M.AgentID, List[M.AgentID]]
                 ] = None,
                 kwargs: Optional[Dict] = None):
        if not callable(entry_point):
            raise error.Error(
                "AgentSpec entry_point must be callable with signature "
                "Callable[[M.POSGModel, M.AgentID, Dict], BasePolicy]"
            )
        self.id = id
        self.entry_point = entry_point

        if valid_agent_ids is None:
            valid_agent_ids = []
        elif isinstance(valid_agent_ids, int):
            valid_agent_ids = [valid_agent_ids]
        self.valid_agent_ids = valid_agent_ids

        self._kwargs = {} if kwargs is None else kwargs

        match = POLICY_ID_RE.search(id)
        if not match:
            raise error.Error(
                f"Attempted to register malformed policy ID: {id}. (Currently "
                f"all IDs must be of the form {POLICY_ID_RE.pattern}.)"
            )
        self._policy_id = match.group(1)
        self._policy_version = match.group(2)

        if "/" in id:
            self.env_id, _ = id.split("/")
        else:
            self.env_id = None

    def make(self,
             model: M.POSGModel,
             agent_id: M.AgentID,
             **kwargs) -> BasePolicy:
        """Instantiates an instance of the agent."""
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)

        if self.valid_agent_ids and agent_id not in self.valid_agent_ids:
            raise error.Error(
                f"Attempted to initialize policy with ID={id} with invalid "
                f"agent ID '{agent_id}'. Valid agent IDs for this policy are: "
                f"{self.valid_agent_ids}."
            )

        agent_pi = self.entry_point(model, agent_id, self.id, **_kwargs)

        # Make the agent policy aware of which spec it came from.
        spec = copy.deepcopy(self)
        spec._kwargs = _kwargs
        agent_pi.spec = spec

        return agent_pi

    def __repr__(self):
        return f"PolicySpec({self.id})"


class PolicyRegistry:
    """Register for policies by ID."""

    def __init__(self):
        self.policy_specs: Dict[str, PolicySpec] = {}

    def make(self,
             id: str,
             model: M.POSGModel,
             agent_id: M.AgentID,
             **kwargs) -> BasePolicy:
        """Make policy with given ID."""
        if len(kwargs) > 0:
            logger.info(
                "Making new policy: %s (model=%s agent_id=%s kwargs=%s)",
                id, model, agent_id, kwargs
            )
        else:
            logger.info(
                "Making new policy: %s (model=%s agent_id=%s)",
                id, model, agent_id
            )
        spec = self.spec(id)
        return spec.make(model, agent_id, **kwargs)

    def all(self) -> Sequence[PolicySpec]:
        """Get all registered PolicySpecs."""
        return self.policy_specs.values()

    def spec(self, id: str) -> PolicySpec:
        """Get PolicySpec with given ID."""
        try:
            match = POLICY_ID_RE.search(id)
        except TypeError as ex:
            raise TypeError(f"{ex}: {id=}")

        if not match:
            raise error.Error(
                "Attempted to look up malformed policy ID: "
                f"{id.encode('utf-8')}. (Currently all IDs must be of the form"
                f" {POLICY_ID_RE.pattern}.)"
            )
        try:
            return self.policy_specs[id]
        except KeyError:
            raise error.Unregistered(f"No registered policy with id: {id}")

    def register(self, id: str, entry_point: PolicyEntryPoint, **kwargs):
        """Register policy in register."""
        self.register_spec(PolicySpec(id, entry_point, kwargs))

    def register_spec(self, policy_spec: PolicySpec):
        """Register policy spec in register."""
        if policy_spec.id in self.policy_specs:
            logger.warn(f"Overriding policy {spec.id}")
        self.policy_specs[policy_spec.id] = policy_spec

    def all_for_env(self,
                    env_id: str,
                    include_generic_policies: bool = True) -> List[PolicySpec]:
        """Get all PolicySpecs that are associated with given environment.

        If `include_generic_policies` is True then will also return policies
        that are valid for all environments (e.g. the random-v0 policy)
        """
        return [
            pi_spec for pi_spec in self.all()
            if (
                pi_spec.env_id == env_id
                or (include_generic_policies and pi_spec.env_id is None)
            )
        ]


# Global registry that all implemented policies are added too
# Policies are registered when posggym-agents library is loaded
registry = PolicyRegistry()


def register(id: str, entry_point: PolicyEntryPoint, **kwargs):
    """Register a policy with posggym-agents."""
    return registry.register(id, entry_point, **kwargs)


def register_spec(spec: PolicySpec):
    """Register a policy spec with posggym-agents."""
    return registry.register_spec(spec)


def make(id: str,
         model: M.POSGModel,
         agent_id: M.AgentID,
         **kwargs) -> BasePolicy:
    """Create a policy according to the given ID, model and agent id."""
    return registry.make(id, model, agent_id, **kwargs)


def spec(id: str) -> PolicySpec:
    """Get the specification of the policy with given ID."""
    return registry.spec(id)
