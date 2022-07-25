from convlab2.dialog_agent import PipelineAgent
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.nlg.template.multiwoz import TemplateNLG

from simulator.agenda_policy import AgendaPolicy

class SIMULATOR(PipelineAgent):
    def __init__(self, simulator_config):
        nlu = BERTNLU(mode='sys',
                      config_file='multiwoz_sys_context.json',
                      model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_sys_context.zip')
        dst = None
        policy = AgendaPolicy(simulator_config["max_turn"], simulator_config["max_initiative"])
        nlg = TemplateNLG(is_user=True, mode=simulator_config["template_mode"])

        self.name = 'user'
        self.opponent_name = 'sys'
        self.nlu = nlu
        self.dst = dst
        self.policy = policy
        self.nlg = nlg

        self.turn_count = 0
        self.history = []
        self.log = []

    def init_session(self):
        self.turn_count = 0
        self.history = []
        self.log = []

        self.nlu.init_session()
        self.policy.init_session()
        self.nlg.init_session()

    def response(self, observation):
        system_response = observation
        user_utterance = super().response(system_response)
        system_action = self.input_action.copy()
        user_action = self.output_action.copy()

        self.log.append({
            "turn_id": self.turn_count,
            "system_response": system_response,
            "nlu": {"system_action": system_action},
            "policy": {"user_atcion": user_action},
            "nlg": {"user_utterance": user_utterance}
        })
        self.turn_count += 1
        return user_utterance