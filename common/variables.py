import yaml

with open("config/config_file.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# Algo Params
PROB_EPSILON = cfg['algo_params']['prob_epsilon']
PERIODS = cfg['algo_params']['periods']
NESTED_MC_ITERS = cfg['algo_params']['nested_mc_iters']
MC_EPISODES = cfg['algo_params']['mc_episodes']
MAX_STEPS = cfg['algo_params']['max_steps']
EXPLORE_EPS = cfg['algo_params']['explore_eps']

# SS policy
H_COST = cfg['policy_params']['h_cost']
B_PENALTY = cfg['policy_params']['b_penalty']
SMALL_S = cfg['policy_params']['small_s']
BIG_S = cfg['policy_params']['big_s']
N_SUPPLIERS = cfg['policy_params']['n_suppliers']
BACKORDER_MAX = cfg['policy_params']['backorder_max']

# MDP Environment
LAMBDA = cfg['mdp_env_params']['lambda']
STOCK_BOUND = cfg['mdp_env_params']['stock_bound']
ACTION_SIZE = cfg['mdp_env_params']['action_size']
DISCOUNT_FAC = cfg['mdp_env_params']['discount_fac']

