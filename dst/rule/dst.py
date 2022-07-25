import json
from convlab2.policy.vector.vector_multiwoz import MultiWozVector
from convlab2.dst.rule.multiwoz import RuleDST

from dst import AbstractDST, DSTOutput
from util import FixedMultiWozVector

class MyRuleDST(AbstractDST):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "dst"
        assert self.name == "rule"
        self.rule_dst = RuleDST()
        self.vector = FixedMultiWozVector()

    @property
    def module_state_dim(self):
        return self.vector.state_dim

    def module_state_vector(self):
        state_vector = self.vector.state_vectorize(self.rule_dst.state)
        return state_vector

    def init_session(self):
        self.rule_dst.init_session()

    def replace_state(self, key, value):
        if key == "state":
            self.rule_dst.state = value
        elif key in self.rule_dst.state:
            self.rule_dst.state[key] = value
        else:
            raise KeyError("{} is not in rule dst keys ({})".format(key, self.rule_dst.keys()))

    def append_history(self, data):
        self.rule_dst.state["history"].append(data)

    def update(self, action):
        return DSTOutput(state=self.rule_dst.update(action))