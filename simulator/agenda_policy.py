from convlab2.policy.policy import Policy
from convlab2.policy.rule.multiwoz.policy_agenda_multiwoz import UserPolicyAgendaMultiWoz

class AgendaPolicyCore(UserPolicyAgendaMultiWoz):
    def __init__(self, max_turn, max_initiative):
        super().__init__()
        self.max_turn = max_turn * 2
        self.max_initiative = max_initiative

class AgendaPolicy(Policy):

    def __init__(self, max_turn, max_initiative, is_train=False, character='usr'):
        self.is_train = is_train
        self.character = character

        self.policy = AgendaPolicyCore(max_turn, max_initiative)

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        return self.policy.predict(state)

    def init_session(self, **kwargs):
        """
        Restore after one session
        """
        self.policy.init_session(**kwargs)

    def is_terminated(self):
        return self.policy.is_terminated()

    def get_reward(self):
        return self.policy.get_reward()

    def get_goal(self):
        return self.policy.get_goal()