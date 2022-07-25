from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator

class Evaluator(MultiWozEvaluator):
    def __init__(self, max_turn, reward_type):
        super().__init__()
        self.max_turn = max_turn
        if reward_type not in ["task_complete", "task_success"]:
            raise NotImplementedError("Reward type <{}> is not implemented.".format(reward_type))
        self.reward_type = reward_type

    def evaluate(self, user_agent):
        session_over = user_agent.is_terminated()

        if self.reward_type == "task_complete":
            reward = user_agent.policy.get_reward()
            if reward != -1:
                reward /= 2
        
        elif self.reward_type == "task_success":
            reward = self.get_reward()

        elif self.reward_type == "task_success_fixed":
            reward = self.get_reward()
            if reward > 5 and not session_over:
                reward = 5

        return session_over, reward