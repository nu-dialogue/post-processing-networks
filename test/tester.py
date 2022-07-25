import json
import os
import math
import pandas as pd
import torch.multiprocessing as mp
from system import SYSTEM
from simulator import SIMULATOR
from util import Session, Evaluator, DOMAINS, get_logger
logger = get_logger(__name__)
from test.utils import (
    load_system_config, make_system,
    save_table, print_score, plot)

class Tester:
    def __init__(self, sys_agent: SYSTEM, user_agent: SIMULATOR,
                 iteration_id, total_dialogs, process_num, max_turn, log_dpath):
        self.sys_agent = sys_agent
        self.user_agent = user_agent
        self.iteration_id = iteration_id

        self.total_dialogs = total_dialogs
        self.process_num = process_num
        self.max_turn = max_turn
        self.dialogs_per_process = math.ceil(self.total_dialogs / self.process_num)
        
        self.log_dpath = log_dpath

    def _results_sampler(self, pid, queue, evt, sys_agent: SYSTEM, user_agent: SIMULATOR):
        results = []
        for i in range(self.dialogs_per_process):
            evaluator = Evaluator(max_turn=self.max_turn,
                                  reward_type="task_success")
            session = Session(mode="test",
                              sys_agent=sys_agent,
                              user_agent=user_agent,
                              evaluator=evaluator,
                              iteration_id=self.iteration_id,
                              process_id=pid,
                              episode_id=i,
                              log_dpath=self.log_dpath)
            session.init_session()

            system_response = ""
            for t in range(self.max_turn+1):
                user_response, session_over, reward, system_response = session.next_turn(system_response)
                if session_over:
                    break
            
            log = session.save_log()
            del log["user_dialog"], log["system_dialog"]
            for domain in DOMAINS:
                log[f"domain_{domain}"] = domain in log["initial_goal"]
            results.append(log)
        queue.put([pid, results])
        evt.wait()

    def sample_results(self):
        self.sys_agent.test_iteration()

        queue = mp.Queue()
        evt = mp.Event()
        processes = []
        for pid in range(self.process_num):
            processes.append(mp.Process(target=self._results_sampler,
                                        args=(pid, queue, evt, self.sys_agent, self.user_agent)))
            processes[-1].daemon = True
            processes[-1].start()

        results = []
        for _ in range(self.process_num):
            pid_, results_ = queue.get()
            results += results_
        del pid_, results_
        evt.set()

        for process in processes:
            process.join()
        
        return results[:self.total_dialogs]

    def test(self):
        results = self.sample_results()
        return results

def test_single_system(sys_agent: SYSTEM, user_agent: SIMULATOR, test_single_config):
    sys_agent = sys_agent
    user_agent = user_agent

    test_single_id = test_single_config["test_single_id"]
    iteration_id = test_single_config["iteration_id"]
    total_dialogs = test_single_config["total_dialogs"]
    process_num = test_single_config["process_num"]
    max_turn = test_single_config["max_turn"]
    dialogs_per_process = math.ceil(total_dialogs / process_num)

    if not os.path.exists(os.path.dirname(test_single_config["test_config_fpath"])):
        os.makedirs(os.path.dirname(test_single_config["test_config_fpath"]))
    json.dump(test_single_config, open(test_single_config["test_config_fpath"], "w"), indent=4)
    
    log_dpath = test_single_config["log_dpath"]
    if not os.path.exists(log_dpath):
        os.makedirs(log_dpath)
    result_table_dpath = test_single_config["result_table_dpath"]
    if not os.path.exists(result_table_dpath):
        os.makedirs(result_table_dpath)

    tester = Tester(sys_agent=sys_agent, user_agent=user_agent,
                    iteration_id=iteration_id, total_dialogs=total_dialogs,
                    process_num=process_num, max_turn=max_turn, log_dpath=log_dpath)
    results = tester.test()
    iteration_df = pd.DataFrame(results)
    save_table(iteration_df, result_table_dpath=result_table_dpath, iteration_id=iteration_id)

def test_all_system(user_agent: SIMULATOR, test_all_config):
    test_all_id = test_all_config["test_all_id"]
    resume_rl_train_id = test_all_config["resume_rl_train_id"]

    module_list = test_all_config["module_list"]
    base_system_config = load_system_config(resume_rl_train_dpath=test_all_config["resume_rl_train_dpath"])
    user_agent = user_agent

    dialogs_per_iteration = test_all_config["dialogs_per_iteration"]
    process_num = test_all_config["process_num"]
    max_turn = test_all_config["max_turn"]
    dialogs_per_process = math.ceil(dialogs_per_iteration / process_num)

    if not os.path.exists(os.path.dirname(test_all_config["test_all_config_fpath"])):
        os.makedirs(os.path.dirname(test_all_config["test_all_config_fpath"]))
        json.dump( test_all_config, open(test_all_config["test_all_config_fpath"], "w"), indent=4)

    ppn_config_dpath = test_all_config["ppn_config_dpath"]
    if not os.path.exists(ppn_config_dpath):
        os.makedirs(ppn_config_dpath)

    log_dpath = test_all_config["log_dpath"]
    if not os.path.exists(log_dpath):
        os.makedirs(log_dpath)
    
    result_table_dpath = test_all_config["result_table_dpath"]
    if not os.path.exists(result_table_dpath):
        os.makedirs(result_table_dpath)
    
    result_figure_dpath = test_all_config["result_figure_dpath"]
    if not os.path.exists(result_figure_dpath):
        os.makedirs(result_figure_dpath)

    print_metrics = ["task_success", "book_rate", "inform_F1", "turn"]
    plot_metrics = ["task_complete", "task_success", "book_rate", "inform_F1", "inform_precision", "inform_recall", "turn"]

    results = []
    for i_id in [0]:
        logger.info("="*50)
        logger.info(f"Model Iteration: {i_id}")
        system = make_system(base_system_config=base_system_config,
                                ppn_config_dpath=ppn_config_dpath,
                                resume_rl_train_id=resume_rl_train_id,
                                iteration_id=i_id)
        logger.info("<TEST>")
        tester = Tester(sys_agent=system, user_agent=user_agent,
                        iteration_id=i_id, total_dialogs=dialogs_per_iteration,
                        process_num=process_num, max_turn=max_turn, log_dpath=log_dpath)
        results_ = tester.test()
        iteration_df = pd.DataFrame(results_)
        save_table(iteration_df, result_table_dpath=result_table_dpath, iteration_id=i_id)
        print_score(iteration_df, metrics=print_metrics)
        results += results_

    plot(total_df=pd.DataFrame(results),
            metrics=plot_metrics,
            result_figure_dpath=result_figure_dpath)