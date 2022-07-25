import numpy as np

from nlg.sclstm.fixed_sclstm import FixedSCLSTM
from nlg import AbstractNLG, NLGOutput

class MySCLSTMNLG(AbstractNLG):
    def __init__(self, module_type, module_config) -> None:
        super().__init__(module_type, module_config)
        assert self.type == "nlg"
        assert self.name == "sclstm"
        self.sclstm_nlg = FixedSCLSTM()
        self.beamsize_list = [1, 5, 10]
        self.default_beamsize = 5

    @property
    def module_state_dim(self) -> int:
        return len(self.beamsize_list)

    def module_state_vector(self) -> np.ndarray:
        mode_onehot = np.zeros(len(self.beamsize_list))
        mode_onehot[self.beamsize_list.index(self.default_beamsize)] = 1
        return mode_onehot

    def init_session(self) -> None:
        self.sclstm_nlg.init_session()

    def generate(self, dialog_acts: list) -> dict:
        responses = self.sclstm_nlg.generate_with_beamsizes(meta=dialog_acts, beamsize_list=[self.default_beamsize])

        return NLGOutput(system_response=responses[self.default_beamsize], system_responses=responses)
    