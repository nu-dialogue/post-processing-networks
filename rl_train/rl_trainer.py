import math
import os
import json
import random
import pandas as pd
import torch.multiprocessing as mp

from util import Evaluator, Session, DOMAINS, get_logger
logger = get_logger(__name__)
from system import SYSTEM
from simulator import SIMULATOR
from test.utils import save_table, print_score, plot
class RLTrainer:
    def __init__(self, sys_agent: SYSTEM, user_agent: SIMULATOR, rl_train_config: dict) -> None:
        self.sys_agent = sys_agent
        self.user_agent = user_agent

        self.rl_train_id = rl_train_config["rl_train_id"]
    
        self.total_timesteps = rl_train_config["total_timesteps"]
        self.timesteps_per_batch = rl_train_config["timesteps_per_batch"]
        self.process_num = rl_train_config["process_num"]
        self.timesteps_per_process = math.ceil(self.timesteps_per_batch / self.process_num)

        self.max_timesteps_per_episode = rl_train_config["max_timesteps_per_episode"]
        self.reward_type = rl_train_config["reward_type"]
        self.iterations_per_model_save = rl_train_config["iterations_per_model_save"]

        self.selection_strategy = rl_train_config["selection_strategy"]
        self.bc_schedule = rl_train_config["bc_schedule"]
        self.__rotation_indicator = 0
        self.__updatable_ppns = []
        self.__bc_update_timers = {}
        if self.sys_agent.nlu_ppn is not None:
            self.__updatable_ppns.append("nlu_ppn")
            self.__bc_update_timers["nlu_ppn"] = {"remaining": 0, "period": rl_train_config["initial_bc_update_period"]}
        if self.sys_agent.dst_ppn is not None:
            self.__updatable_ppns.append("dst_ppn")
            self.__bc_update_timers["dst_ppn"] = {"remaining": 0, "period": rl_train_config["initial_bc_update_period"]}
        if self.sys_agent.policy_ppn is not None:
            self.__updatable_ppns.append("policy_ppn")
            self.__bc_update_timers["policy_ppn"] = {"remaining": 0, "period": rl_train_config["initial_bc_update_period"]}
        if self.sys_agent.nlg_ppn is not None:
            self.__updatable_ppns.append("nlg_ppn")
            self.__bc_update_timers["nlg_ppn"] = {"remaining": 0, "period": rl_train_config["initial_bc_update_period"]}
        self.max_bc_update_period = rl_train_config["max_bc_update_period"]
        
        # self.test_dialogs_per_iteration = rl_train_config["test_dialogs_per_iteration"]
        # self.test_dialogs_per_process = math.ceil(self.test_dialogs_per_iteration / self.process_num)

        self.rl_train_log_dpath = rl_train_config["rl_train_log_dpath"]
        if not os.path.exists(self.rl_train_log_dpath):
            os.makedirs(self.rl_train_log_dpath)
        
        # self.test_log_dpath = rl_train_config["test_log_dpath"]
        # if not os.path.exists(self.test_log_dpath):
        #     os.makedirs(self.test_log_dpath)

        self.rl_train_results = []
        self.rl_train_result_figure_dpath = rl_train_config["rl_train_result_figure_dpath"]
        if not os.path.exists(self.rl_train_result_figure_dpath):
            os.makedirs(self.rl_train_result_figure_dpath)
        
        self.rl_train_result_table_dpath = rl_train_config["rl_train_result_table_dpath"]
        if not os.path.exists(self.rl_train_result_table_dpath):
            os.makedirs(self.rl_train_result_table_dpath)

        json.dump(rl_train_config, open(rl_train_config["rl_train_config_fpath"], "w"), indent=4)
        
        self.print_metrics = ["task_success", "book_rate", "inform_F1", "turn"]
        self.plot_metrics = ["task_complete", "task_success", "book_rate", "inform_F1", "inform_precision", "inform_recall", "turn"]

    def __select_ppo_updating_ppns(self):
        updating_ppns = []
        if self.selection_strategy == "all":
            for ppn_type in self.__updatable_ppns:
                updating_ppns.append(ppn_type)
        elif self.selection_strategy == "random":
            updating_num = random.randint(1, len(self.__updatable_ppns))
            for ppn_type in random.sample(self.__updatable_ppns, k=updating_num):
                updating_ppns.append(ppn_type)
        elif self.selection_strategy == "rotation":
            updating_ppns.append( self.__updatable_ppns[self.__rotation_indicator] )
            self.__rotation_indicator = (self.__rotation_indicator + 1) % len(self.__updatable_ppns)
        else:
            raise Exception("{} is unknown selection_strategy.".format(self.selection_strategy))
        for updating_ppn in updating_ppns:
            self.__bc_update_timers[updating_ppn]["remaining"] -= 1
        return updating_ppns

    def __select_bc_updating_ppns(self):
        updating_ppns = []
        if self.bc_schedule == "gradually":
            for ppn_type, timer in self.__bc_update_timers.items():
                if timer["remaining"] == 0:
                    updating_ppns.append(ppn_type)
                    if timer["period"] < self.max_bc_update_period:
                        # If max bc has not yet been reached, set another timer to perform bc
                        timer["period"] += 1
                        logger.info(f"Incremented {ppn_type}'s bc update period ({timer['period']-1} -> {timer['period']})")
                        timer["remaining"] = timer["period"]
                        logger.info(f"Set {ppn_type}'s bc update timer to {timer}")
                elif timer["remaining"] < 0:
                    logger.info(f"{ppn_type} is not bc updated any more.")
                else:
                    continue
        elif self.bc_schedule == "initially":
            if self.sampled_iterations == 0:
                updating_ppns = list(self.__bc_update_timers)
            else:
                logger.info(f"All ppns are not bc updated because of <initilally> mode.")
        elif self.bc_schedule == "no_update":
            logger.info(f"All ppns are not bc updated because of <no_update> mode.")
        else:
            raise Exception("{} is unknown bc_schedule.".format(self.bc_schedule))
        return updating_ppns

    def _ppo_rollouts_sampler(self, pid, queue, evt, sys_agent, user_agent):
        sampled_timesteps, sampled_episode = 0, 0
        results = []
        while sampled_timesteps < self.timesteps_per_process:

            evaluator = Evaluator(max_turn=self.max_timesteps_per_episode,
                                  reward_type=self.reward_type)
            session = Session(mode="rl_train",
                              sys_agent=sys_agent,
                              user_agent=user_agent,
                              evaluator=evaluator,
                              iteration_id=self.sampled_iterations,
                              process_id=pid,
                              episode_id=sampled_episode,
                              log_dpath=self.rl_train_log_dpath)
            session.init_session()

            system_response = ""
            for t in range(self.max_timesteps_per_episode+1):
                user_response, session_over, reward, system_response = session.next_turn(system_response)
                if session_over:
                    break
                
            sampled_timesteps += t
            sampled_episode += 1
            log = session.save_log()
            del log["user_dialog"], log["system_dialog"]
            for domain in DOMAINS:
                log[f"domain_{domain}"] = domain in log["initial_goal"]
            results.append(log)
 
        queue.put([pid, sampled_timesteps, sys_agent.get_ppo_rollouts(), results])
        evt.wait()

    def rl_train(self):
        logger.info("<TRAIN>")
        self.sys_agent.train_iteration()
        queue = mp.Queue()
        evt = mp.Event()
        processes = []
        for pid in range(self.process_num):
            processes.append(mp.Process(target=self._ppo_rollouts_sampler,
                                        args=(pid, queue, evt, self.sys_agent, self.user_agent)))
            processes[-1].daemon = True
            processes[-1].start()
        results = []
        for _ in range(self.process_num):
            pid_, sampled_timesteps_, rollouts_by_ppn_, results_ = queue.get()
            self.sampled_timesteps += sampled_timesteps_
            self.sys_agent.merge_ppo_rollouts(rollouts_by_ppn_)
            results += results_
        del pid_, sampled_timesteps_, rollouts_by_ppn_, results_
        evt.set()
        for process in processes:
            process.join()
        
        iteration_df = pd.DataFrame(results)
        save_table(iteration_df=iteration_df,
                   result_table_dpath=self.rl_train_result_table_dpath,
                   iteration_id=self.sampled_iterations)
        print_score(iteration_df=iteration_df,
                    metrics=self.print_metrics)

        self.rl_train_results += results
        self.sys_agent.ppo_update(sampled_timesteps=self.sampled_timesteps,
                                  total_timesteps=self.total_timesteps,
                                  ppo_updating_ppns=self.__select_ppo_updating_ppns())
        if (self.iterations_per_model_save > 0) and \
            (self.sampled_iterations % self.iterations_per_model_save == 0):
            self.sys_agent.save_ppn(self.sampled_iterations)

    # def test(self):
    #     logger.info("<TEST>")
    #     if not self.test_dialogs_per_iteration:
    #         logger.info("Skip Test.")
    #         return
    #     tester = Tester(sys_agent=self.sys_agent, user_agent=self.user_agent,
    #                     iteration_id=self.sampled_iterations, total_dialogs=self.test_dialogs_per_iteration,
    #                     process_num=self.process_num, max_turn=self.max_timesteps_per_episode,
    #                     log_dpath=self.test_log_dpath, result_table_dpath=None)
    #     tester.test(print_score=True)
    
    def behavior_clone(self):
        logger.info("<BEHAVIOR CLONE>")
        bc_updating_ppns = self.__select_bc_updating_ppns()
        if bc_updating_ppns:
            self.sys_agent.bc_update(bc_updating_ppns=bc_updating_ppns)
        else:
            logger.info("Skip Behavior Clone Update.")

    def run(self):
        self.sampled_iterations = 0
        self.sampled_timesteps = 0
        
        if self.bc_schedule != "no_update":
            self.sys_agent.prepare_bc_updator()

        self.behavior_clone()
        self.sys_agent.save_ppn(0)

        logger.info("Start RL Train.")
        while self.sampled_timesteps < self.total_timesteps:
            self.sampled_iterations += 1
            logger.info("="*50)
            logger.info(f"Iteration: {self.sampled_iterations}")
            logger.info(f"Sampled timesteps so far: {self.sampled_timesteps}")
            self.rl_train()
            # self.test()
            self.behavior_clone()
        logger.info("End RL Train.")
        plot(total_df=pd.DataFrame(self.rl_train_results),
             metrics=self.plot_metrics,
             result_figure_dpath=self.rl_train_result_figure_dpath)