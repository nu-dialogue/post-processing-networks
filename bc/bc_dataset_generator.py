import json
import math
import os
import pandas as pd
import torch.multiprocessing as mp
from util import Evaluator, Session, get_logger
logger = get_logger(__name__)

class BehaviorCloneDatasetGenerator:
    def __init__(self, sys_agent, user_agent, bcd_generate_config):
        self.sys_agent = sys_agent
        self.user_agent = user_agent

        self.total_timesteps = bcd_generate_config["total_timesteps"]
        self.max_timesteps_per_episode = bcd_generate_config["max_timesteps_per_episode"]
        self.process_num = bcd_generate_config["process_num"]
        self.timesteps_per_process =  math.ceil(self.total_timesteps / self.process_num)

        self.bc_dataset_dpath = bcd_generate_config["bc_dataset_dpath"]
        if not os.path.exists(self.bc_dataset_dpath):
            os.makedirs(self.bc_dataset_dpath)

        self.log_dpath = bcd_generate_config["log_dpath"]
        if not os.path.exists(self.log_dpath):
            os.makedirs(self.log_dpath)

        json.dump(bcd_generate_config, open(bcd_generate_config["bcd_generate_config_fpath"], "w"), indent=4)

    def _bc_dataset_sampler(self, pid, queue, evt, sys_agent, user_agent):
        sampled_timesteps, sampled_episode = 0, 0
        results = []
        while sampled_timesteps < self.timesteps_per_process:

            evaluator = Evaluator(max_turn=self.max_timesteps_per_episode,
                                  reward_type="task_success")
            session = Session(mode="bcd_generate",
                              sys_agent=sys_agent,
                              user_agent=user_agent,
                              evaluator=evaluator,
                              iteration_id=0,
                              process_id=pid,
                              episode_id=sampled_episode,
                              log_dpath=self.log_dpath)
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
            results.append(log)

        queue.put([pid, sys_agent.get_bc_dataset(), results, sampled_timesteps])
        evt.wait()

    def generate(self):
        logger.info("<GENERATE BEHAVIOR CLONING DATASET>")
        self.sys_agent.bcd_generate()
        queue = mp.Queue()
        evt = mp.Event()
        processes = []
        for pid in range(self.process_num):
            processes.append(mp.Process(target=self._bc_dataset_sampler,
                                        args=(pid, queue, evt, self.sys_agent, self.user_agent)))
            processes[-1].daemon = True
            processes[-1].start()

        results = []
        sampled_timesteps = 0
        for _ in range(self.process_num):
            pid_, bc_dataset_by_ppn_, results_, sampled_timesteps_ = queue.get()
            self.sys_agent.merge_bc_dataset(bc_dataset_by_ppn_)
            results += results_
            sampled_timesteps += sampled_timesteps_
        del pid_, bc_dataset_by_ppn_, sampled_timesteps_
        evt.set()

        for process in processes:
            process.join()
        return self.sys_agent.get_bc_dataset(), results, sampled_timesteps

    def process(self, bc_dataset_by_ppn, results, sampled_timesteps):
        df = pd.DataFrame(results)
        scores = {key: score for key, score in df[["task_success", "inform_F1", "book_rate", "turn"]].mean().items() }
        data_info = {"total_size": sampled_timesteps,
                     "total_dialogs": len(results),
                     "scores": scores}
        data_info_fpath = os.path.join(self.bc_dataset_dpath, "data_info.json")
        json.dump(data_info, open(data_info_fpath, "w"), indent=4)

        for ppn_type, bc_dataset in bc_dataset_by_ppn.items():
            bc_dataset.save(bcd_dpath=self.bc_dataset_dpath)

    def run(self):
        bc_dataset_by_ppn, results, sampled_timesteps = self.generate()
        self.process(bc_dataset_by_ppn, results, sampled_timesteps)