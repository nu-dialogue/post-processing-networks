from copy import deepcopy
import os
import json

def exclude_dst_history(log):
    """Delete system.dst["state"]["history"]"""
    log = deepcopy(log)
    for turn_id in range(len(log)):
        try:
            if "history" in log[turn_id]["dst"]["state"]:
                del log[turn_id]["dst"]["state"]["history"]
            if "state" in log[turn_id]["dst_ppn"] and "history" in log[turn_id]["dst_ppn"]["state"]:
                del log[turn_id]["dst_ppn"]["state"]["history"]
        except KeyError:
            pass
    return log

class Session:
    """
    Class that manages the interaction between the system and the user simulator
    Implemented based on convlab2.dialog_agent.BiSession
    """
    def __init__(self, mode, sys_agent, user_agent, evaluator, iteration_id, process_id, episode_id, log_dpath):
        self.mode = mode
        self.sys_agent = sys_agent
        self.user_agent = user_agent
        self.evaluator = evaluator
        self.iteration_id = iteration_id
        self.process_id = process_id
        self.episode_id = episode_id
        self.log_fpath = os.path.join(log_dpath, "{}-{}-{}.json".format(iteration_id, process_id, episode_id))

    def init_session(self):
        self.sys_agent.init_session()
        self.user_agent.init_session()
        goal = self.user_agent.policy.get_goal()
        self.evaluator.add_goal(goal)

        self.final_goal = goal
        self.initial_goal = deepcopy(goal)

        self.turn = 0

        self.common_reward_history = []
        self.reward_history_by_module = []

    def next_turn(self, last_system_response):
        user_response = self.user_agent.response(last_system_response)

        self.evaluator.add_sys_da(self.user_agent.get_in_da())
        self.evaluator.add_usr_da(self.user_agent.get_out_da())

        session_over, reward = self.evaluator.evaluate(self.user_agent)

        if self.turn > 0:
            self.common_reward_history.append(reward)
        self.sys_agent.save_reward(reward, session_over) # Processing at turn 0 is ignored in sys_agent.

        system_response = self.sys_agent.response(user_response, final=session_over)
        # The sys_agent handles the last turn to end the episode.
        self.turn += 1
        return user_response, session_over, reward, system_response

    def save_log(self):
        prec, rec, F1 = self.evaluator.inform_F1()
        log_data = {
            "mode": self.mode,
            "iteration_id": self.iteration_id,
            "process_id": self.process_id,
            "episode_id": self.episode_id,
            "initial_goal": self.initial_goal,
            "final_goal": self.final_goal,
            "task_complete": self.user_agent.policy.policy.goal.task_complete(),
            "task_success": self.evaluator.task_success(),
            "book_rate": self.evaluator.book_rate(),
            "inform_F1": F1,
            "inform_precision": prec,
            "inform_recall": rec,
            "turn": self.turn,
            "common_reward_history":self.common_reward_history,
            "user_dialog": self.user_agent.log,
            "system_dialog": exclude_dst_history(self.sys_agent.log)
        }

        json.dump(log_data, open(self.log_fpath, "w"), indent=4)
        
        return log_data

    def train(self):
        loss = self.sys_agent.update_ppn()
        return loss