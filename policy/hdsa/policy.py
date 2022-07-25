import numpy as np
from convlab2.policy.hdsa.multiwoz import HDSA

from policy import AbstractPolicy, PolicyOutput

class MyHDSAPolicy(AbstractPolicy):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        self.type == "policy"
        self.name == "hdsa"
        self.hdsa_policy = HDSA()
        self.num_act_label = 44 # from hdsa predictor

    @property
    def module_state_dim(self):
        return self.num_act_label

    def module_state_vector(self) -> np.ndarray:
        assert self.current_act.shape[0] == self.module_state_dim
        return self.current_act

    def init_session(self) -> None:
        self.hdsa_policy.init_session()
        self.current_act = np.zeros(self.num_act_label)

    def predict(self, state: dict):
        act, kb = self.hdsa_policy.predictor.predict(state)
        response = self.hdsa_policy.generator.generate(state, act, kb)
        self.current_act = act[0]
        return PolicyOutput(system_action=response)