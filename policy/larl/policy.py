import numpy as np
from convlab2.policy.larl.multiwoz import LaRL

from policy import AbstractPolicy, PolicyOutput

class MyLaRLPolicy(AbstractPolicy):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "policy"
        assert self.name == "larl"
        self.larl_policy = LaRL()

    @property
    def module_state_dim(self) -> int:
        return 0

    def module_state_vector(self) -> np.ndarray:
        return np.empty(0)

    def init_session(self) -> None:
        self.larl_policy.init_session()

    def predict(self, state: dict):
        response = self.larl_policy.predict(state)
        return PolicyOutput(system_action=response)
        