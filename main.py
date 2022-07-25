import sys
import numpy as np
import random
import torch
import torch.multiprocessing as mp

from util import ROOT_DPATH
sys.path.append(ROOT_DPATH)

from arguments import (
    get_args,
    args2system_config,
    args2simulator_config,
    args2rl_train_config,
    args2test_all_config,
    args2test_single_config,
    args2bcd_generate_config
)

def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bcd_generate(args):
    from system import SYSTEM
    from simulator import SIMULATOR
    from bc import BehaviorCloneDatasetGenerator

    bcd_generate_config = args2bcd_generate_config(args)
    system_config = args2system_config(args)
    simulator_config = args2simulator_config(args)

    system = SYSTEM(system_config=system_config)
    simulator = SIMULATOR(simulator_config=simulator_config)
    bcd_generator = BehaviorCloneDatasetGenerator(sys_agent=system,
                                                  user_agent=simulator,
                                                  bcd_generate_config=bcd_generate_config)
    bcd_generator.run()

def rl_train(args):
    from system import SYSTEM
    from simulator import SIMULATOR
    from rl_train import RLTrainer

    rl_train_config = args2rl_train_config(args)
    system_config = args2system_config(args)
    simulator_config = args2simulator_config(args)

    system = SYSTEM(system_config=system_config)
    simulator = SIMULATOR(simulator_config=simulator_config)
    trainer = RLTrainer(sys_agent=system,
                        user_agent=simulator,
                        rl_train_config=rl_train_config)
    trainer.run()

def test_single(args):
    from system import SYSTEM
    from simulator import SIMULATOR
    from test import test_single_system

    test_single_config = args2test_single_config(args)
    system_config = args2system_config(args)
    simulator_config = args2simulator_config(args)
    system = SYSTEM(system_config=system_config)
    simulator = SIMULATOR(simulator_config=simulator_config)

    test_single_system(sys_agent=system,
                       user_agent=simulator,
                       test_single_config=test_single_config)

def test_all(args):
    from simulator import SIMULATOR
    from test import test_all_system

    test_all_config = args2test_all_config(args)
    simulator_config = args2simulator_config(args)
    simulator = SIMULATOR(simulator_config=simulator_config)
    
    test_all_system(user_agent=simulator, test_all_config=test_all_config)

if __name__ == "__main__":
    args = get_args()

    set_seed(args)
    mp.set_start_method('spawn')

    if args.run_type == "bcd_generate":
        bcd_generate(args)
    elif args.run_type == "rl_train":
        rl_train(args)
    elif args.run_type == "test_single":
        test_single(args)
    elif args.run_type == "test_all":
        test_all(args)
    else:
        raise NotImplementedError("run type <{}>".format(args.run_type))