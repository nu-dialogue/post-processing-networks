import numpy as np

from ppn import AbstractPPN
from nlg import NLGOutput, NLGPPNOutput
from nlg.utils import ResponseEncoder
from nlg.sclstm.nlg import MySCLSTMNLG

class MySCLSTMNLGPPN(AbstractPPN):
    def __init__(self, module: MySCLSTMNLG, system_state_dim: int):
        raise DeprecationWarning("SCLSTM PPN is deprecated.")
        super().__init__(module, system_state_dim)

    def _prepare_vocab(self):
        self.__default_beamsize = self.module.default_beamsize
        self.__beamsize_list = self.module.beamsize_list
        self.__response_encoder = ResponseEncoder()
        self.__vec_dim = self.__response_encoder.vec_dim

    def _set_model_params(self):
        self.model_params.update({
            "obs_dim":self.__vec_dim*len(self.__beamsize_list) + self.system_state_dim,
            "act_dim": len(self.__beamsize_list)
        })

    def _make_observation(self, module_output, system_state):
        responses = module_output.system_responses
        resp_vec_list = [self.__response_encoder.vectorize(responses[mode]) for mode in self.__beamsize_list]
        observation = np.concatenate( resp_vec_list + [system_state] )
        return observation, self.__beamsize_list.index(self.__default_beamsize), {"responses": responses}

    def _devectorize(self, ppn_output_vec, **items):
        responses = items["responses"]
        top_mode = self.__beamsize_list[ppn_output_vec]
        top_response = responses[top_mode]
        return NLGPPNOutput(system_response=top_response, top1=top_mode)

    def _through_module_output(self, module_output: NLGOutput):
        return module_output