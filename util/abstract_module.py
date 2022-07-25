from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

class AbstractModule(ABC):
    def __init__(self, module_type, module_config) -> None:
        self.type = module_type
        self.name = module_config["module_name"]
        self.ppn_config = module_config["ppn_config"]

    @property
    @abstractmethod
    def module_state_dim(self) -> int:
        """Number of dimensions of each module's state information"""
        pass

    @abstractmethod
    def module_state_vector(self) -> np.ndarray:
        """Number of dimensions of each module's output"""
        pass

    @abstractmethod
    def init_session(self) -> None:
        pass

class AbstractModuleOutput(ABC):
    def dcopy(self):
        return deepcopy(self)
    def as_dict(self):
        return deepcopy(self.__dict__)