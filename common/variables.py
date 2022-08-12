import yaml
import platform
import numpy as np

config_path = "config/config_file.yaml" if "larkin" in platform.node() else "config/config_file.yaml"

with open(config_path, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# Algo Params
PROB_EPSILON = cfg['algo_params']['prob_epsilon']
PERIODS = cfg['algo_params']['periods']
NESTED_MC_ITERS = cfg['algo_params']['nested_mc_iters']
MC_EPISODES = cfg['algo_params']['mc_episodes']
MAX_STEPS = cfg['algo_params']['max_steps']
EXPLORE_EPS = cfg['algo_params']['explore_eps']
DEBUG_BOOL = cfg['algo_params']['debug_bool']
FIXED_LEARN_RATE = cfg['algo_params']['fixed_learn_rate']

# SS policy
H_COST = cfg['policy_params']['h_cost']
B_PENALTY = cfg['policy_params']['b_penalty']
SMALL_S = cfg['policy_params']['small_s']
BIG_S = cfg['policy_params']['big_s']
N_SUPPLIERS = cfg['policy_params']['n_suppliers']
BACKORDER_MAX = cfg['policy_params']['backorder_max']
INVEN_LIMIT = cfg['policy_params']['inven_limit']

# MDP Environment
LAMBDA = cfg['mdp_env_params']['lambda']
MAX_INVEN = cfg['mdp_env_params']['max_inven']
ACTION_SIZE = cfg['mdp_env_params']['action_size']
DISCOUNT_FAC = cfg['mdp_env_params']['discount_fac']

PROCUREMENT_COST_VEC = np.array(cfg['mdp_env_params']['procurement_cost_vec'])
SUPPLIER_LEAD_TIMES_VEC = np.array(cfg['mdp_env_params']['supplier_lead_times_vec'])
ON_TIMES = np.array(cfg['mdp_env_params']['on_times'])
OFF_TIMES = np.array([np.Inf, np.Inf]) if cfg['mdp_env_params']['off_times'] == "no_disrup" else np.array(cfg['mdp_env_params']['off_times'])

# Evaluation Params
SAFE_FACTOR = cfg['eval_params']['safe_factor'] if 'eval_params' in cfg and 'safe_factor' in cfg['eval_params'] else 1.0
N_VISIT_LIM = cfg['eval_params']['n_visit_limit'] if 'eval_params' in cfg and 'safe_factor' in cfg['eval_params'] else 2
SUB_EVAL_PERIODS = cfg['eval_params']['sub_eval_periods'] if 'eval_params' in cfg and 'sub_eval_periods' in cfg['eval_params'] else 1.0
SUB_NESTED_MC_ITER = cfg['eval_params']['sub_nested_mc_iter'] if 'eval_params' in cfg and 'sub_nested_mc_iter' in cfg['eval_params'] else 1.0

# Dual Index
DI_DEL_RNG = cfg['dual_index']['delta_cand_range'] if 'dual_index' in cfg and 'delta_cand_range' in cfg['dual_index'] else 20
DI_SF_FAC = cfg['dual_index']['di_safety_factor'] if 'dual_index' in cfg and 'di_safety_factor' in cfg['dual_index'] else 8

##
MODEL_ARGS_DIC = cfg

MAX_INVEN_LP = cfg['lp_config']['max_inven_lp'] if 'lp_config' in cfg and 'max_inven_lp' in cfg['lp_config'] else 8
BACKORDER_MAX_LP = cfg['lp_config']['backorder_max_lp'] if 'lp_config' in cfg and 'backorder_max_lp' in cfg['lp_config'] else -3
ACTION_SIZE_LP = cfg['lp_config']['action_size_lp'] if 'lp_config' in cfg and 'action_size_lp' in cfg['lp_config'] else 4
