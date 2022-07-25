import os
from convlab2.policy.rule.multiwoz.rule_based_multiwoz_bot import RuleBasedMultiwozBot

from policy import AbstractPolicy, PolicyOutput
from util import FixedMultiWozVector

class MyRulePolicy(AbstractPolicy):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "policy"
        assert self.name == "rule"
        self.rule_policy = RuleBasedMultiwozBot()
        self.vector = FixedMultiWozVector()

    @property
    def module_state_dim(self) -> int:
        return self.vector.da_dim
    
    def module_state_vector(self):
        da_vec = self.vector.action_vectorize(self.predicted_action)
        assert da_vec.shape[0] == self.module_state_dim
        return da_vec

    def predict(self, state: dict):
        _state_vec = self.vector.state_vectorize(state)
        self.predicted_action = self.rule_policy.predict(state)
        return PolicyOutput(system_action=self.predicted_action)

    def init_session(self) -> None:
        self.rule_policy.init_session()
        self.predicted_action = []