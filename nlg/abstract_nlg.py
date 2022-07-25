from copy import deepcopy
from abc import abstractmethod
from util import AbstractModule, AbstractModuleOutput

class NLGOutput(AbstractModuleOutput):
    def __init__(self, system_response: str, system_responses: dict) -> None:
        self.system_response = system_response
        self.system_responses = system_responses
    def get_output_response(self):
        return self.system_response

class NLGPPNOutput(NLGOutput):
    def __init__(self, system_response: str, top1: str) -> None:
        super().__init__(system_response=system_response, system_responses=None)
        del self.system_responses
        self.top1 = top1

class AbstractNLG(AbstractModule):
    def __init__(self, module_type, module_config) -> None:
        assert module_type == "nlg"
        super().__init__(module_type, module_config)

    @abstractmethod
    def generate(self, dialog_acts: list) -> dict:
        pass