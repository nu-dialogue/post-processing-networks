from abc import abstractmethod
from copy import deepcopy
from util import AbstractModule, AbstractModuleOutput

class DSTOutput(AbstractModuleOutput):
    def __init__(self, state: dict):
        self.state = state
    def get_state(self):
        return deepcopy(self.state)

class DSTPPNOutput(DSTOutput):
    def __init__(self, state: dict, deleted_bs: list, added_rs:list, deleted_rs: list) -> None:
        super().__init__(state=state)
        self.deleted_bs = deleted_bs
        self.added_rs = added_rs
        self.deleted_rs = deleted_rs

class AbstractDST(AbstractModule):
    def __init__(self, module_type, module_config) -> None:
        assert module_type == "dst"
        super().__init__(module_type, module_config)

    @abstractmethod
    def replace_state(self, key: str, value: any) -> None:
        pass

    @abstractmethod
    def append_history(self, data: list):
        pass

    @abstractmethod
    def update(self, action: list) -> DSTOutput:
        pass
