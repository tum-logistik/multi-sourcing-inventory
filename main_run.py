from environment.SourcingEnv import *
import numpy as np

if __name__ == '__main__':
    print("run")

    sourcingEnv = SourcingEnv(
        lambda_arrival = 5, # or 10
        procurement_cost_vec = np.array([3, 1, 2]),
        supplier_lead_times_vec = np.array([0.5, 0.75, 1.0]),
        on_times = np.array([3, 1, 2]), 
        off_times = np.array([0.3, 1, 1.5]))
    
    history = []

    for i in range(1999):
        random_action = [np.random.randint(0, 4) for x in range(sourcingEnv.n_suppliers)] 

        next_state, event, event_index, probs, supplier_index = sourcingEnv.step(random_action)
        sum_check =  np.sum(probs)
        history.append([next_state, event, i, probs])
    
    print(sum_check)