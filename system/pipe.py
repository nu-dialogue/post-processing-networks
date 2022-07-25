class Pipe:
    def __init__(self, turn_id):
        self.turn_id = turn_id
        self.input_utterance = None
        self.input_action = None
        self.state = None
        self.output_action = None
        self.output_response = None

        self.nlu_output = None
        self.nlu_ppn_output = None
        self.dst_output = None
        self.dst_ppn_output = None
        self.policy_output = None
        self.policy_ppn_output = None
        self.nlg_output = None 
        self.nlg_ppn_output = None

    def start(self, user_utterance):
        self.input_utterance = user_utterance

    def step(self,
             nlu_output=None, nlu_ppn_output=None, dst_output=None, dst_ppn_output=None,
             policy_output=None, policy_ppn_output=None, nlg_output=None, nlg_ppn_output=None):
        
        if nlu_output is not None:
            self.nlu_output = nlu_output.dcopy()
            self.input_action = self.nlu_output.get_input_action()

        elif nlu_ppn_output is not None:
            self.nlu_ppn_output = nlu_ppn_output.dcopy()
            self.input_action = self.nlu_ppn_output.get_input_action()

        elif dst_output is not None:
            self.dst_output = dst_output.dcopy()
            self.state = self.dst_output.get_state()

        elif dst_ppn_output is not None:
            self.dst_ppn_output = dst_ppn_output.dcopy()
            self.state = self.dst_ppn_output.get_state()

        elif policy_output is not None:
            self.policy_output = policy_output.dcopy()
            self.output_action = self.policy_output.get_output_action()

        elif policy_ppn_output is not None:
            self.policy_ppn_output = policy_ppn_output.dcopy()
            self.output_action = self.policy_ppn_output.get_output_action()

        elif nlg_output is not None:
            self.nlg_output = nlg_output.dcopy()
            self.output_response = self.nlg_output.get_output_response()

        elif nlg_ppn_output is not None:
            self.nlg_ppn_output = nlg_ppn_output.dcopy()
            self.output_response = self.nlg_ppn_output.get_output_response()

        else:
            raise Exception("Enter any of the outputs.")

    def skip(self, nlu_output=False, nlg_output=False):
        if nlu_output:
            self.input_action = self.input_utterance
        if nlg_output:
            self.output_response = self.output_action

    def get_log(self):
        log = {"turn_id": self.turn_id}

        log["input_utterance"] = self.input_utterance

        if self.nlu_output is not None:
            log["nlu"] = self.nlu_output.as_dict()

        if self.nlu_ppn_output is not None:
            log["nlu_ppn"] = self.nlu_ppn_output.as_dict()
        
        if self.dst_output is not None:
            log["dst"] = self.dst_output.as_dict()
        
        if self.dst_ppn_output is not None:
            log["dst_ppn"] = self.dst_ppn_output.as_dict()
        
        if self.policy_output is not None:
            log["policy"] = self.policy_output.as_dict()
        
        if self.policy_ppn_output is not None:
            log["policy_ppn"] = self.policy_ppn_output.as_dict()
        
        if self.nlg_output is not None:
            log["nlg"] = self.nlg_output.as_dict()
        
        if self.nlg_ppn_output is not None:
            log["nlg_ppn"] = self.nlg_ppn_output.as_dict()

        log["output_response"] = self.output_response
        
        return log