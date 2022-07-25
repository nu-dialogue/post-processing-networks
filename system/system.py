import json
from system.system_state import SystemState
from system.pipe import Pipe
from system.utils import create_module, create_ppn
from util import get_logger
logger = get_logger(__name__)
class SYSTEM:
    def __init__(self, system_config: dict):
        self.name = "sys"
        self.opponent_name = "user"
        
        self.nlu = create_module(module_type="nlu", system_config=system_config)
        self.dst = create_module(module_type="dst", system_config=system_config)
        self.policy = create_module(module_type="policy", system_config=system_config)
        self.nlg = create_module(module_type="nlg", system_config=system_config)

        self.system_state = SystemState(nlu=self.nlu, dst=self.dst, policy=self.policy, nlg=self.nlg)

        self.nlu_ppn = create_ppn(module=self.nlu, system_state_dim=self.system_state.dim)
        self.dst_ppn = create_ppn(module=self.dst, system_state_dim=self.system_state.dim)
        self.policy_ppn = create_ppn(module=self.policy, system_state_dim=self.system_state.dim)
        self.nlg_ppn = create_ppn(module=self.nlg, system_state_dim=self.system_state.dim)

    def bcd_generate(self):
        if self.nlu_ppn is not None:
            self.nlu_ppn.bcd_generate()
        if self.dst_ppn is not None:
            self.dst_ppn.bcd_generate()
        if self.policy_ppn is not None:
            self.policy_ppn.bcd_generate()
        if self.nlg_ppn is not None:
            self.nlg_ppn.bcd_generate()

    def prepare_bc_updator(self):
        if self.nlu_ppn is not None:
            self.nlu_ppn.prepare_bc_updator()
        if self.dst_ppn is not None:
            self.dst_ppn.prepare_bc_updator()
        if self.policy_ppn is not None:
            self.policy_ppn.prepare_bc_updator()
        if self.nlg_ppn is not None:
            self.nlg_ppn.prepare_bc_updator()

    def train_iteration(self):
        if self.nlu_ppn is not None:
            self.nlu_ppn.train_iteration()
        if self.dst_ppn is not None:
            self.dst_ppn.train_iteration()
        if self.policy_ppn is not None:
            self.policy_ppn.train_iteration()
        if self.nlg_ppn is not None:
            self.nlg_ppn.train_iteration()

    def test_iteration(self):
        if self.nlu_ppn is not None:
            self.nlu_ppn.test_iteration()
        if self.dst_ppn is not None:
            self.dst_ppn.test_iteration()
        if self.policy_ppn is not None:
            self.policy_ppn.test_iteration()
        if self.nlg_ppn is not None:
            self.nlg_ppn.test_iteration()

    def init_session(self):
        if self.nlu is not None:
            self.nlu.init_session()
        self.dst.init_session()
        self.policy.init_session()
        if self.nlg is not None:
            self.nlg.init_session()
        
        self.dst.append_history([self.name, 'null'])

        self.system_state.init_session()
        if self.nlu is not None:
            self.system_state.update("nlu", self.nlu.module_state_vector())
        self.system_state.update("dst", self.dst.module_state_vector())
        self.system_state.update("policy", self.policy.module_state_vector())
        if self.nlg is not None:
            self.system_state.update("nlg", self.nlg.module_state_vector())

        self.turn_count = 0
        self.history = []
        self.log = []

        if self.nlu_ppn is not None:
            self.nlu_ppn.sample_episode()
        if self.dst_ppn is not None:
            self.dst_ppn.sample_episode()
        if self.policy_ppn is not None:
            self.policy_ppn.sample_episode()
        if self.nlg_ppn is not None:
            self.nlg_ppn.sample_episode()

    def save_reward(self, reward, done):
        # Receive user utterance and rewards resulting from the system's actions in the previous turn
        rewards_by_ppn = {}
        if self.turn_count == 0:
            return rewards_by_ppn

        if self.nlu_ppn is not None:
            rewards_by_ppn["nlu"] = self.nlu_ppn.save_reward(reward, done)
        if self.dst_ppn is not None:
            rewards_by_ppn["dst"] = self.dst_ppn.save_reward(reward, done)
        if self.policy_ppn is not None:
            rewards_by_ppn["policy"] = self.policy_ppn.save_reward(reward, done)
        if self.nlg_ppn is not None:
            rewards_by_ppn["nlg"] = self.nlg_ppn.save_reward(reward, done)
        return rewards_by_ppn

    def get_ppo_rollouts(self):
        ppo_rollouts_by_ppn = {}
        if self.nlu_ppn is not None:
            ppo_rollouts_by_ppn["nlu_ppn"] = self.nlu_ppn.get_ppo_rollouts()
        if self.dst_ppn is not None:
            ppo_rollouts_by_ppn["dst_ppn"] = self.dst_ppn.get_ppo_rollouts()
        if self.policy_ppn is not None:
            ppo_rollouts_by_ppn["policy_ppn"] = self.policy_ppn.get_ppo_rollouts()
        if self.nlg_ppn is not None:
            ppo_rollouts_by_ppn["nlg_ppn"] = self.nlg_ppn.get_ppo_rollouts()
        return ppo_rollouts_by_ppn

    def merge_ppo_rollouts(self, ppo_rollouts_by_ppn):
        if self.nlu_ppn is not None:
            self.nlu_ppn.merge_ppo_rollouts(ppo_rollouts_by_ppn["nlu_ppn"])
        if self.dst_ppn is not None:
            self.dst_ppn.merge_ppo_rollouts(ppo_rollouts_by_ppn["dst_ppn"])
        if self.policy_ppn is not None:
            self.policy_ppn.merge_ppo_rollouts(ppo_rollouts_by_ppn["policy_ppn"])
        if self.nlg_ppn is not None:
            self.nlg_ppn.merge_ppo_rollouts(ppo_rollouts_by_ppn["nlg_ppn"])

    def ppo_update(self, sampled_timesteps, total_timesteps, ppo_updating_ppns):
        loss = {"nlu_ppn": None, "dst_ppn": None, "policy_ppn": None, "nlg_ppn": None}
        if self.nlu_ppn is not None and "nlu_ppn" in ppo_updating_ppns:
            loss["nlu_ppn"] = self.nlu_ppn.ppo_update(sampled_timesteps, total_timesteps)
        if self.dst_ppn is not None and "dst_ppn" in ppo_updating_ppns:
            loss["dst_ppn"] = self.dst_ppn.ppo_update(sampled_timesteps, total_timesteps)
        if self.policy_ppn is not None and "policy_ppn" in ppo_updating_ppns:
            loss["policy_ppn"] = self.policy_ppn.ppo_update(sampled_timesteps, total_timesteps)
        if self.nlg_ppn is not None and "nlg_ppn" in ppo_updating_ppns:
            loss["nlg_ppn"] = self.nlg_ppn.ppo_update(sampled_timesteps, total_timesteps)
        return loss.copy()

    def bc_update(self, bc_updating_ppns):
        loss = {"nlu_ppn": None, "dst_ppn": None, "policy_ppn": None, "nlg_ppn": None}
        if self.nlu_ppn is not None and "nlu_ppn" in bc_updating_ppns:
            loss["nlu_ppn"] = self.nlu_ppn.bc_update()
        if self.dst_ppn is not None and "dst_ppn" in bc_updating_ppns:
            loss["dst_ppn"] = self.dst_ppn.bc_update()
        if self.policy_ppn is not None and "policy_ppn" in bc_updating_ppns:
            loss["policy_ppn"] = self.policy_ppn.bc_update()
        if self.nlg_ppn is not None and "nlg_ppn" in bc_updating_ppns:
            loss["nlg_ppn"] = self.nlg_ppn.bc_update()
        return loss.copy()

    def get_bc_dataset(self):
        bc_dataset_by_ppn = {}
        if self.nlu_ppn is not None:
            bc_dataset_by_ppn["nlu_ppn"] = self.nlu_ppn.get_bc_dataset()
        if self.dst_ppn is not None:
            bc_dataset_by_ppn["dst_ppn"] = self.dst_ppn.get_bc_dataset()
        if self.policy_ppn is not None:
            bc_dataset_by_ppn["policy_ppn"] = self.policy_ppn.get_bc_dataset()
        if self.nlg_ppn is not None:
            bc_dataset_by_ppn["nlg_ppn"] = self.nlg_ppn.get_bc_dataset()
        return bc_dataset_by_ppn

    def merge_bc_dataset(self, bc_dataset_by_ppn):
        if self.nlu_ppn is not None:
            self.nlu_ppn.merge_bc_dataset(bc_dataset_by_ppn["nlu_ppn"])
        if self.dst_ppn is not None:
            self.dst_ppn.merge_bc_dataset(bc_dataset_by_ppn["dst_ppn"])
        if self.policy_ppn is not None:
            self.policy_ppn.merge_bc_dataset(bc_dataset_by_ppn["policy_ppn"])
        if self.nlg_ppn is not None:
            self.nlg_ppn.merge_bc_dataset(bc_dataset_by_ppn["nlg_ppn"])

    def save_ppn(self, iterations_so_far):
        if self.nlu_ppn: self.nlu_ppn.save_model(iterations_so_far)
        if self.dst_ppn: self.dst_ppn.save_model(iterations_so_far)
        if self.policy_ppn: self.policy_ppn.save_model(iterations_so_far)
        if self.nlg_ppn: self.nlg_ppn.save_model(iterations_so_far)

    def response(self, user_utterance, final=False):
        self.dst.replace_state("terminated", final)
        pipe = Pipe(turn_id=self.turn_count)
        
        try:
            # >>> Prepare for current turn >>> 
            pipe.start(user_utterance=user_utterance)
            self.dst.append_history([self.opponent_name, pipe.input_utterance])
            self.history.append([self.opponent_name, pipe.input_utterance])
            # <<< Prepare for current turn <<<


            # >>> NLU >>>
            if self.nlu is not None:
                pipe.step(nlu_output=self.nlu.predict(observation=user_utterance,
                                                      context=[utt for _, utt in self.history[:-1]]))
                self.system_state.update("nlu", self.nlu.module_state_vector())
                if self.nlu_ppn is not None:
                    pipe.step(nlu_ppn_output=self.nlu_ppn.postprocess(module_output=pipe.nlu_output,
                                                                      system_state=self.system_state.latest(),
                                                                      final=final))
            else:
                pipe.skip(nlu_output=True)
            # <<< NLU <<<


            # >>> DST >>>
            self.dst.replace_state("user_action", pipe.input_action)
            pipe.step(dst_output=self.dst.update(pipe.input_action))
            self.system_state.update("dst", self.dst.module_state_vector())
            if self.dst_ppn is not None:
                pipe.step(dst_ppn_output=self.dst_ppn.postprocess(module_output=pipe.dst_output,
                                                                  system_state=self.system_state.latest(),
                                                                  final=final))
                self.dst.replace_state("state", pipe.state)
            # <<< DST <<<


            # >>> Policy >>>
            pipe.step(policy_output=self.policy.predict(pipe.state))
            self.system_state.update("policy", self.policy.module_state_vector())
            if self.policy_ppn is not None:
                pipe.step(policy_ppn_output=self.policy_ppn.postprocess(module_output=pipe.policy_output,
                                                                        system_state=self.system_state.latest(),
                                                                        final=final))
            # <<< Policy <<<


            # >>> NLG >>>
            if self.nlg is not None:
                pipe.step(nlg_output=self.nlg.generate(pipe.output_action))
                self.system_state.update("nlg", self.nlg.module_state_vector())
                if self.nlg_ppn is not None:
                    pipe.step(nlg_ppn_output=self.nlg_ppn.postprocess(module_output=pipe.nlg_output,
                                                                      system_state=self.system_state.latest(),
                                                                      final=final))
            else:
                pipe.skip(nlg_output=True)
            # <<< NLG <<<


            # >>> Prepare for next turn >>>
            self.dst.replace_state("system_action", pipe.output_action)
            self.dst.append_history([self.name, pipe.output_response])
            self.history.append([self.name, pipe.output_response])
            self.turn_count += 1
            # <<< Prepare for next turn <<<

            self.log.append(pipe.get_log())
            return pipe.output_response
            
        except Exception as e:
            # logger.info("history::")
            # logger.info(json.dumps(self.history, indent=4))
            # logger.info("current_turn::")
            # logger.info(json.dumps(pipe.get_log(), indent=4))
            raise e
  
