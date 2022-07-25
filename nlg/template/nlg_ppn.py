import numpy as np

from ppn import AbstractPPN
from nlg import NLGOutput, NLGPPNOutput
from nlg.template.nlg import MyTemplateNLG
from nlg.utils import ResponseEncoder

class MyTemplateNLGPPN(AbstractPPN):
    def __init__(self, module: MyTemplateNLG, system_state_dim: int) -> None:
        raise DeprecationWarning("TemplateNLG PPN is deprecated.")
        super().__init__(module=module, system_state_dim=system_state_dim)
    
    def _prepare_vocab(self):
        self.__default_mode = self.module.default_mode
        self.__mode_list = self.module.mode_list
        self.__response_encoder = ResponseEncoder()
        self.__vec_dim = self.__response_encoder.vec_dim

    def _set_model_params(self):
        self.model_params.update({
            "obs_dim":self.__vec_dim*len(self.__mode_list) + self.system_state_dim,
            "act_dim": len(self.__mode_list)
        })

    def _verctorize(self, module_output: NLGOutput):
        responses = module_output.system_responses
        resp_vec_list = [self.__response_encoder.vectorize(responses[mode]) for mode in self.__mode_list]
        response_vec = np.concatenate( resp_vec_list )
        # mode_vec = self.__mode_list.index(self.__default_mode)
        return response_vec, {"responses": responses}

    def _devectorize(self, processed_module_output_vec, **items):
        responses = items["responses"]
        top_mode = self.__mode_list[processed_module_output_vec]
        top_response = responses[top_mode]
        return NLGPPNOutput(system_response=top_response, top1=top_mode)

    def _through_module_output(self, module_output: NLGOutput):
        return module_output