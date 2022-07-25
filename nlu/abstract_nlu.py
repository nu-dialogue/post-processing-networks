from abc import abstractmethod
from copy import deepcopy
from util import AbstractModule, AbstractModuleOutput

class NLUOutput(AbstractModuleOutput):
    def __init__(self, user_action: list) -> None:
        self.user_action = user_action
    def get_input_action(self):
        return deepcopy(self.user_action)

class NLUPPNOutput(NLUOutput):
    def __init__(self, user_action: list, processed_da: list) -> None:
        super().__init__(user_action=user_action)
        self.processed_da = processed_da

class AbstractNLU(AbstractModule):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)

    @abstractmethod
    def predict(self, utterance: str, context: list) -> NLUOutput:
        pass