import os
import json
from copy import deepcopy
import argparse
from distutils.util import strtobool

from util import ROOT_DPATH
OUTPUTS_DPATH = os.path.join(ROOT_DPATH, "outputs")
from modules import MODULE_DICT

def _add_common(parser):
    group = parser.add_argument_group("Common arguments")
    group.add_argument("--run_type", choices=["bcd_generate", "rl_train", "test_single", "test_all"], required=True,
                       help="Type of process you want to run. Select from the following:\n"
                            "'bcd_generate': generate data for behavior cloning\n"
                            "'rl_train': train ppns with dialogue simulations and reinforcement learning\n"
                            "'test_single': test trained ppn models at a specific iteration")
    group.add_argument("--run_id", type=str, required=True,
                       help="An arbitrary string for the run ID.\n"
                            "Specify a unique string since it will be used for\n"
                            "the output directory name and identification ID of\n"
                            "the result data and the training model.")
    group.add_argument("--random_seed", type=int, default=999,
                       help="Random seed.")
    group.add_argument("--process_num", type=int, default=8,
                       help="Number of processes used for the dialogue simulation.\n"
                       "Note that the random seed set above does not work in sub-processes.")

def _add_nlu(parser):
    # NLU
    group = parser.add_argument_group("NLU arguments")
    group.add_argument("--nlu_name", choices=list(MODULE_DICT["nlu"].keys())+[""],
                       help="Name of the model to be used as the NLU module.")
    # PPN
    group.add_argument("--nlu_ppn_use", type=strtobool, default="False",
                       help="Whether to use PPN_NLU.")
    group.add_argument("--nlu_ppn_resume_rl_train_id", type=str, default="",
                       help="Run ID of trained PPN_NLU you want to use.")
    group.add_argument("--nlu_ppn_resume_rl_iteration_id", type=str, default="",
                       help="Which iteration of trained PPN_NLU's checkpoint to be load.")
    group.add_argument("--nlu_ppn_model_hid_dim", type=int, default=128,
                       help="Number of hidden units of MLP in PPN_NLU.")
    group.add_argument("--nlu_ppn_model_type", choices=["discrete", "multi_binary"], default="multi_binary",
                       help="Output format of MLP in PPN_NLU.")
    group.add_argument("--nlu_ppn_activation_type", choices=["relu", "tanh"], default="relu",
                       help="Activation function of MLP in PPN_NLU.")
    group.add_argument("--nlu_ppn_initialize_weight", type=strtobool, default="False",
                       help="Orthogonal initialization of MLP weights in PPN_NLU.")
    group.add_argument("--nlu_ppn_use_system_state", type=strtobool, default="True",
                       help="Whether PPN_NLU uses system state (s_All)")

def _add_dst(parser):
    group = parser.add_argument_group("DST arguments")
    # DST
    group.add_argument("--dst_name", choices=list(MODULE_DICT["dst"].keys())+[""],
                       help="Name of the model to be used as the DST module.")
    # PPN
    group.add_argument("--dst_ppn_use", type=strtobool, default="False",
                       help="Whether to use PPN_DST.")
    group.add_argument("--dst_ppn_resume_rl_train_id", type=str, default="",
                       help="Run ID of trained PPN_DST you want to use.")
    group.add_argument("--dst_ppn_resume_rl_iteration_id", type=str, default="",
                       help="Which iteration of trained PPN_DST's checkpoint to be load.")
    group.add_argument("--dst_ppn_model_hid_dim", type=int, default=128,
                       help="Number of hidden units of MLP in PPN_DST.")
    group.add_argument("--dst_ppn_model_type", choices=["discrete", "multi_binary"], default="multi_binary",
                       help="Output format of MLP in PPN_DST.")
    group.add_argument("--dst_ppn_activation_type", choices=["relu", "tanh"], default="relu",
                       help="Activation function of MLP in PPN_DST.")
    group.add_argument("--dst_ppn_initialize_weight", type=strtobool, default="False",
                       help="Orthogonal initialization of MLP weights in PPN_DST.")
    group.add_argument("--dst_ppn_use_system_state", type=strtobool, default="True",
                       help="Whether PPN_DST uses system state (s_All)")

def _add_policy(parser):
    group = parser.add_argument_group("Policy arguments")
    # Policy
    group.add_argument("--policy_name", choices=list(MODULE_DICT["policy"].keys())+[""],
                       help="Name of the model to be used as the Policy module.")
    # PPN
    group.add_argument("--policy_ppn_use", type=strtobool, default="False",
                       help="Whether to use PPN_Policy.")
    group.add_argument("--policy_ppn_resume_rl_train_id", type=str, default="",
                       help="Run ID of trained PPN_Policy you want to use.")
    group.add_argument("--policy_ppn_resume_rl_iteration_id", type=str, default="",
                       help="Which iteration of trained PPN_Policy's checkpoint to be load.")
    group.add_argument("--policy_ppn_model_hid_dim", type=int, default=128,
                       help="Number of hidden units of MLP in PPN_Policy.")
    group.add_argument("--policy_ppn_model_type", choices=["discrete", "multi_binary"], default="multi_binary",
                       help="Output format of MLP in PPN_Policy.")
    group.add_argument("--policy_ppn_activation_type", choices=["relu", "tanh"], default="relu",
                       help="Activation function of MLP in PPN_Policy.")
    group.add_argument("--policy_ppn_initialize_weight", type=strtobool, default="False",
                       help="Orthogonal initialization of MLP weights in PPN_Policy.")
    group.add_argument("--policy_ppn_use_system_state", type=strtobool, default="True",
                       help="Whether PPN_Policy uses system state (s_All)")

def _add_nlg(parser):
    group = parser.add_argument_group("NLG arguments")
    # NLG
    group.add_argument("--nlg_name", choices=list(MODULE_DICT["nlg"].keys())+[""],
                       help="Name of the model to be used as the NLG module.")
    # PPN
    group.add_argument("--nlg_ppn_use", type=strtobool, default="False",
                       help="Whether to use PPN_NLG.\n***Note that PPN_NLG is not implemented.***")
    group.add_argument("--nlg_ppn_resume_rl_train_id", type=str, default="",
                       help="Run ID of trained PPN_NLG you want to use.")
    group.add_argument("--nlg_ppn_resume_rl_iteration_id", type=str, default="",
                       help="Which iteration of trained PPN_NLG's checkpoint to be load.")
    group.add_argument("--nlg_ppn_model_hid_dim", type=int, default=128,
                       help="Number of hidden units of MLP in PPN_NLG.")
    group.add_argument("--nlg_ppn_model_type", choices=["discrete", "multi_binary"], default="discrete",
                       help="Output format of MLP in PPN_NLG.")
    group.add_argument("--nlg_ppn_activation_type", choices=["relu", "tanh"], default="relu",
                       help="Activation function of MLP in PPN_NLG.")
    group.add_argument("--nlg_ppn_initialize_weight", type=strtobool, default="False",
                       help="Orthogonal initialization of MLP weights in PPN_NLG.")
    group.add_argument("--nlg_ppn_use_system_state", type=strtobool, default="True",
                       help="Whether PPN_NLG uses system state (s_All)")

def _add_simulator(parser):
    group = parser.add_argument_group("Simulator arguments")
    group.add_argument("--simulator_max_initiative", type=int, default=4,
                       help="Maximum number of slots the simulator inform at one time.\n"
                            "See convlab2 implementation for more detail.")
    group.add_argument("--simulator_template_mode", choices=["manual", "auto", "auto_manual"], default="manual",
                       help="Generation mode of the simulator's template NLG")

def _add_bcd_generate(parser):
    group = parser.add_argument_group("Behavior cloning dataset generation arguments")
    group.add_argument("--bcd_generate_total_timesteps", type=int, default=10000,
                       help="Total number of turns (timesteps) to be sampled")
    group.add_argument("--bcd_generate_max_timesteps_per_episode", type=int, default=20,
                       help="Maximum number of turns (timesteps) in each dialogue (episode)")

def _add_rl_train(parser):
    group = parser.add_argument_group("RL train arguments")
    # RL Trainer
    group.add_argument("--rl_train_total_timesteps", type=int, default=200000,
                       help="Total number of turns (timesteps) used in RL")
    group.add_argument("--rl_train_timesteps_per_batch", type=int, default=1024,
                       help="Number of turns (timesteps) used to update PPNs in one interation via PPO")
    group.add_argument("--rl_train_max_timesteps_per_episode", type=int, default=20,
                       help="Maximum number of turns (timesteps) in each dialogue (episode)")
    group.add_argument("--rl_train_iterations_per_model_save", type=int, default=1,
                       help="Interval of iteration to save PPN models")
    group.add_argument("--rl_train_reward_type", choices=["task_compelte", "task_success", "task_success_fixed"], default="task_success",
                       help="Types of rewards used in RL")
    group.add_argument("--rl_train_selection_strategy", choices=["all", "rotation", "random"], default="random",
                       help="PPN selection strategy. See our paper for more detail.")
    # group.add_argument("--rl_train_test_dialogs_per_iteration", type=int, default=0)

    # Behavior Clone
    group.add_argument("--rl_train_bc_schedule", choices=["no_update", "initially"], default="initially",
                       help="Schedule imitation learning (behavior cloning; BC) during RL.\n"
                            "'no_update': Never perform BC; we have not experimented yet.\n"
                            "'initially': Perform BC only once before RL\n")
    # group.add_argument("--rl_train_initial_bc_update_period", type=int, default=1)
    # group.add_argument("--rl_train_max_bc_update_period", type=int, default=0)
    group.add_argument("--bcd_generate_id", type=str, default="",
                       help="Run ID of data used in BC")
    group.add_argument("--bc_dataset_total_size", type=int, default=10000,
                       help="Number of turns of data used in BC.\nMust be less than the size of the data specified above.")
    group.add_argument("--bc_dataset_train_size_ratio", type=float, default=0.8,
                       help="Percentage of the BC data used for training.\nThe rest of the data is used for validation.")
    group.add_argument("--bc_epoch_num", type=int, default=20,
                       help="Maximum number of epochs in BC")
    group.add_argument("--bc_early_stopping_patience", type=int, default=0,
                       help="Early stopping patience in BC. Validation loss is used for the stop evaluation.")
    group.add_argument("--bc_mini_batch_size", type=int, default=32,
                       help="Minibatch size in BC")
    group.add_argument("--bc_optimizer", choices=["adam", "rmsprop"], default="adam",
                       help="Optimizer used in BC")
    group.add_argument("--bc_learning_rate", type=float, default=1e-3,
                       help="Learning rate used in BC")

    # PPO Model
    group.add_argument("--ppo_epoch_num", type=int, default=5,
                       help="Number of epochs in each iteration of PPO")
    group.add_argument("--ppo_mini_batch_size", type=int, default=32,
                       help="Minibatch size in each iteration of PPO")
    group.add_argument("--ppo_clip_param", type=float, default=0.2,
                       help="PPO clip parameter in PPO")
    group.add_argument("--ppo_value_loss_coef", type=float, default=0.5,
                       help="Value loss coefficient in PPO")
    group.add_argument("--ppo_entropy_coef", type=float, default=1e-2,
                       help="Entropy term coefficient in PPO")
    group.add_argument("--ppo_optimizer", choices=["adam", "rmsprop"], default="adam",
                       help="Optimizer in PPO")
    group.add_argument("--ppo_lr", type=float, default=1e-4,
                       help="Learning rate in PPO")
    group.add_argument("--ppo_eps", type=float, default=1e-8,
                       help="Epsilon of the optimizer in PPO")
    group.add_argument("--ppo_max_grad_norm", type=float, default=1.0,
                       help="Max norm of gradients in PPO")
    group.add_argument("--ppo_gamma", type=float, default=0.99,
                       help="Discount factor for rewards in PPO")
    group.add_argument("--ppo_gae_lambda", type=float, default=0.95,
                       help="Gae lambda parameter in PPO")
    group.add_argument("--ppo_use_clipped_value_loss", type=strtobool, default="True",
                       help="Value loss coefficient in PPO")
    group.add_argument("--ppo_use_linear_lr_decay", type=strtobool, default="True",
                       help="Use a linear schedule on the learning rate in PPO")
    group.add_argument("--ppo_deterministic_train", type=strtobool, default="False",
                       help="If True, select the maximum likelihood action without sampling\n"
                            "the actions according to the probability distribution during RL")

def _add_test_single(parser):
    group = parser.add_argument_group("Test single arguments")
    group.add_argument("--test_single_total_dialogs", type=int, default=1000,
                       help="Number of dialogue simulations to be performed during test_single")
    group.add_argument("--test_single_max_turn", type=int, default=20,
                       help="Maximum number of turns for each dialogue in test_single")

def _add_test_all(parser):
    group = parser.add_argument_group("Test all arguments")
    group.add_argument("--test_all_dialogs_per_iteration", type=int, default=100,
                       help="Number of dialogue simulations to be performed during test_all")
    group.add_argument("--test_all_max_turn", type=int, default=20,
                       help="Maximum number of turns for each dialogue in test_all")

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    _add_common(parser)
    _add_nlu(parser)
    _add_dst(parser)
    _add_policy(parser)
    _add_nlg(parser)
    _add_simulator(parser)
    _add_bcd_generate(parser)
    _add_rl_train(parser)
    _add_test_all(parser)
    _add_test_single(parser)

    args = parser.parse_args()
    return args

def args2system_config(args):
    def make_resume_fpath(run_type, module_type, resume_train_id, resume_epoch_id):
        assert run_type in ["rl_train"]
        resume_fpath = ""
        if bool(resume_train_id) and bool(resume_epoch_id):
            resume_fpath = os.path.join(OUTPUTS_DPATH, run_type, resume_train_id, "models", module_type, f"{resume_epoch_id}.model")
        elif bool(resume_train_id) or bool(resume_epoch_id):
            raise Exception("'resume_train_id' and 'resume_epoch_id' of {} is inconsistent.".format(module_type))
        return resume_fpath

    system_config = {}
    for module_type in MODULE_DICT.keys():
        module_name = getattr(args, f"{module_type}_name")
        ppn_type = "{}_ppn".format(module_type)
        # 1. model params
        model_params = {
            "model_type": getattr(args, f"{ppn_type}_model_type"),
            "obs_dim": None,
            "hid_dim": getattr(args, f"{ppn_type}_model_hid_dim"),
            "act_dim": None,
            "activation_type": getattr(args, f"{ppn_type}_activation_type"),
            "initialize_weight": getattr(args, f"{ppn_type}_initialize_weight"),
            "epoch_num": args.ppo_epoch_num,
            "mini_batch_size": args.ppo_mini_batch_size,
            "clip_param": args.ppo_clip_param,
            "value_loss_coef": args.ppo_value_loss_coef,
            "entropy_coef": args.ppo_entropy_coef,
            "optimizer": args.ppo_optimizer,
            "lr": args.ppo_lr,
            "eps": args.ppo_eps,
            "max_grad_norm": args.ppo_max_grad_norm,
            "gamma": args.ppo_gamma,
            "gae_lambda": args.ppo_gae_lambda,
            "use_clipped_value_loss": args.ppo_use_clipped_value_loss,
            "use_linear_lr_decay": args.ppo_use_linear_lr_decay,
            "deterministic_train": args.ppo_deterministic_train
        }

        # 2. bc config
        bc_dataset_dpath = ""
        if args.bcd_generate_id:
            bc_dataset_dpath = os.path.join(OUTPUTS_DPATH, "bcd_generate", args.bcd_generate_id)
        bc_config = {
            "bcd_generate_id": args.bcd_generate_id,
            "module_combination": {
                "nlu": args.nlu_name,
                "dst": args.dst_name,
                "policy": args.policy_name,
                "nlg": args.nlg_name},
            "bc_dataset_dpath": bc_dataset_dpath,
            "bc_dataset_total_size": args.bc_dataset_total_size,
            "bc_dataset_train_size_ratio": args.bc_dataset_train_size_ratio,
            "epoch_num": args.bc_epoch_num,
            "mini_batch_size": args.bc_mini_batch_size,
            "optimizer": args.bc_optimizer,
            "learning_rate": args.bc_learning_rate,
            "early_stopping_patience": args.bc_early_stopping_patience,
        }

        # 3. ppn config
        ppn_config_fpath = os.path.join(OUTPUTS_DPATH, args.run_type, args.run_id, "ppn_configs", ppn_type+".json")
        trained_model_dpath = ""
        if args.run_type in ["rl_train"]:
            trained_model_dpath = os.path.join( OUTPUTS_DPATH, args.run_type, args.run_id, "models", ppn_type)
        ppn_config = {
            "module_name": module_name,
            "ppn_type": ppn_type,
            "use": getattr(args, f"{ppn_type}_use"), # bool
            "rl_train": args.run_type == "rl_train", # bool
            "run_id": args.run_id,
            "ppn_config_fpath": ppn_config_fpath,
            "trained_model_dpath": trained_model_dpath,
            "resume_rl_train_id": getattr(args, f"{ppn_type}_resume_rl_train_id"),
            "resume_rl_iteration_id": getattr(args, f"{ppn_type}_resume_rl_iteration_id"),
            "resume_rl_trained_model_fpath": make_resume_fpath("rl_train", ppn_type,
                                                               getattr(args, f"{ppn_type}_resume_rl_train_id"),
                                                               getattr(args, f"{ppn_type}_resume_rl_iteration_id")),
            "use_system_state": getattr(args, f"{ppn_type}_use_system_state"),
            "model_params": model_params,
            "bc_config": bc_config
        }

        # 4. module config
        module_config = {
            "module_name": module_name,
            "ppn_config": ppn_config
        }

        system_config[module_type] = deepcopy(module_config)

    return system_config

def args2simulator_config(args):
    if args.run_type == "bcd_generate":
        max_turn = args.bcd_generate_max_timesteps_per_episode
    elif args.run_type == "rl_train":
        max_turn = args.rl_train_max_timesteps_per_episode
    elif args.run_type == "test_all":
        max_turn = args.test_all_max_turn
    elif args.run_type == "test_single":
        max_turn = args.test_single_max_turn
    else:
        raise Exception("simulator can not be used except in 'rl_train' or 'test'")
    simulator_config = {
        "max_turn": max_turn,
        "max_initiative": args.simulator_max_initiative,
        "template_mode": args.simulator_template_mode
    }
    return simulator_config

def args2bcd_generate_config(args):
    outputs_dpath = os.path.join(OUTPUTS_DPATH, args.run_type, args.run_id)
    bcd_generate_config = {
        "bcd_id": args.run_id,
        "random_seed": args.random_seed,
        "module_combination": {
            "nlu": args.nlu_name,
            "dst": args.dst_name,
            "policy": args.policy_name,
            "nlg": args.nlg_name},
        "total_timesteps": args.bcd_generate_total_timesteps,
        "max_timesteps_per_episode": args.bcd_generate_max_timesteps_per_episode,
        "process_num": args.process_num,
        "bc_dataset_dpath": os.path.join(outputs_dpath),
        "log_dpath": os.path.join(outputs_dpath, "log"),
        "bcd_generate_config_fpath": os.path.join(outputs_dpath, "bcd_generate_config.json")
    }
    return bcd_generate_config

def args2rl_train_config(args):
    outputs_dpath = os.path.join(OUTPUTS_DPATH, "rl_train", args.run_id)
    rl_train_config = {
        "rl_train_id": args.run_id,
        "random_seed": args.random_seed,
        "total_timesteps": args.rl_train_total_timesteps,
        "timesteps_per_batch": args.rl_train_timesteps_per_batch,
        "max_timesteps_per_episode": args.rl_train_max_timesteps_per_episode,
        "iterations_per_model_save": args.rl_train_iterations_per_model_save,
        "process_num": args.process_num,
        "reward_type": args.rl_train_reward_type,
        "selection_strategy": args.rl_train_selection_strategy,
        # "test_dialogs_per_iteration": args.rl_train_test_dialogs_per_iteration,
        "rl_train_config_fpath": os.path.join(outputs_dpath, "rl_train_config.json"),
        "rl_train_log_dpath": os.path.join(outputs_dpath, "rl_train_log"),
        "rl_train_result_table_dpath": os.path.join(outputs_dpath, "rl_train_result_tables"),
        "rl_train_result_figure_dpath": os.path.join(outputs_dpath, "rl_train_result_figures"),
        # "test_log_dpath": os.path.join(outputs_dpath, "test_log"),
        "bc_schedule": args.rl_train_bc_schedule,
        "initial_bc_update_period": args.rl_train_initial_bc_update_period,
        "max_bc_update_period": args.rl_train_max_bc_update_period,
    }
    return rl_train_config


def args2test_single_config(args):
    outputs_dpath = os.path.join(OUTPUTS_DPATH, "test_single", args.run_id)
    test_single_config = {
        "test_single_id": args.run_id,
        "random_seed": args.random_seed,
        "iteration_id": "",
        "total_dialogs": args.test_single_total_dialogs,
        "process_num": args.process_num,
        "max_turn": args.test_single_max_turn,
        "test_config_fpath": os.path.join(outputs_dpath, "test_config.json"),
        "log_dpath":os.path.join(outputs_dpath, "test_single_log"),
        "result_table_dpath":os.path.join(outputs_dpath, "test_single_result_tables")
    }
    return test_single_config

def args2test_all_config(args):
    module_list = {module_type: getattr(args, f"{module_type}_name") for module_type in MODULE_DICT.keys()}
    outputs_dpath = os.path.join(OUTPUTS_DPATH, "test_all", args.run_id)
    test_all_config = {
        "test_all_id": args.run_id,
        "random_seed": args.random_seed,
        "module_list": module_list,
        "resume_rl_train_id": args.run_id,
        "resume_rl_train_dpath": os.path.join(OUTPUTS_DPATH, "test_all", args.run_id),
        "max_turn": args.test_all_max_turn,
        "dialogs_per_iteration": args.test_all_dialogs_per_iteration,
        "process_num": args.process_num,
        "test_all_config_fpath": os.path.join(outputs_dpath, "test_all_config.json"),
        "ppn_config_dpath": os.path.join(outputs_dpath, "ppn_configs"),
        "log_dpath": os.path.join(outputs_dpath, "test_all_log"),
        "result_table_dpath": os.path.join(outputs_dpath, "test_all_result_tables"),
        "result_figure_dpath": os.path.join(outputs_dpath, "test_all_result_figures")
    }
    return test_all_config

if __name__ == "__main__":
    args = get_args()
    tmp = {}
    tmp["system_config"] = args2system_config(args)
    if args.run_type == "bcd_generate":
        tmp["bcd_generate"] = args2bcd_generate_config(args)
    elif args.run_type == "rl_train":
        tmp["rl_train_config"] = args2rl_train_config(args)
    elif args.run_type == "test_single":
        tmp["test_single_config"] = args2test_single_config(args)
    elif args.run_type == "test_all":
        tmp["test_all_config"] = args2test_all_config(args)

    json.dump(tmp, open("arguments.json", "w"), indent=4)