import pickle as pkl
from opt.eval_policy import *
from opt.mc_sim import *
import time
from common.variables import *

filename = "msource_value_dic_08-17-2022-17-53-42.pkl"

with open("output/" + filename, 'rb') as f:
    output_obj = pkl.load(f)

value_dic = output_obj["state_value_dic"]
model_params = output_obj["model_params"]
sourcingEnv = output_obj["mdp_env"]

off_times = np.array([np.Inf, np.Inf]) if cfg['mdp_env_params']['off_times'] == "no_disrup" else np.array(cfg['mdp_env_params']['off_times'])

sourcingEnv2 = SourcingEnv(
    lambda_arrival = model_params['mdp_env_params']['lambda'], # or 10
    procurement_cost_vec = np.array(model_params['mdp_env_params']['procurement_cost_vec']),
    supplier_lead_times_vec = np.array(model_params['mdp_env_params']['supplier_lead_times_vec']),
    on_times = np.array(model_params['mdp_env_params']['on_times']), 
    off_times =  off_times) 

s_custom = MState(stock_level = 0, 
    n_suppliers = N_SUPPLIERS, 
    n_backorders = np.array([0, 0]), 
    flag_on_off = np.array([1, 1]))

kwargs = {
    "value_dic": value_dic, 
    "initial_state": s_custom,
    "periods": 10, 
    "periods_val_it": 1,
    "nested_mc_iters": 5,
    "max_stock": BIG_S,
    "discount_fac": DISCOUNT_FAC,
    "h_cost": model_params['policy_params']['h_cost'],
    "b_penalty": model_params['policy_params']['b_penalty'],
    "n_visit_lim": N_VISIT_LIM,
    "default_ss_policy": ss_policy_fastest_supp_backlog,
    "safe_factor": SAFE_FACTOR,
    "sub_eval_periods": SUB_EVAL_PERIODS,
    "sub_nested_mc_iter": SUB_NESTED_MC_ITER,
    "max_stock": 2,
    "approx_eval": True,
    "filename": filename
}

lp_mdp_cost = mc_with_policy(sourcingEnv2, start_state = s_custom, 
    policy_callback=lp_mdp_policy,
    use_tqdm = True,
    **kwargs)

print(np.mean(lp_mdp_cost))