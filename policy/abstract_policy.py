from abc import abstractmethod
from copy import deepcopy

from util import AbstractModule, AbstractModuleOutput

class PolicyOutput(AbstractModuleOutput):
    def __init__(self, system_action: list)-> None:
        self.system_action = system_action
    def get_output_action(self):
        return deepcopy(self.system_action)

class PolicyPPNOutput(PolicyOutput):
    def __init__(self, system_action: list, processed_da: list) -> None:
        super().__init__(system_action=system_action)
        self.processed_da = processed_da

class AbstractPolicy(AbstractModule):
    def __init__(self, module_type, module_config) -> None:
        assert module_type == "policy"
        super().__init__(module_type, module_config)

    @abstractmethod
    def predict(self, state: dict):
        pass