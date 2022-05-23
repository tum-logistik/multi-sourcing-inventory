from environment.SourcingEnv import *
import numpy as np

if __name__ == '__main__':
    print("run")

    sourcingEnv = SourcingEnv(
        lambda_arrival = 100, 
        on_times = np.array([3, 1])/8, 
        off_times = np.array([0.3, 1])/7)
    
    history = []

    for i in range(1999):
        random_action = [np.random.randint(0, 21)] * sourcingEnv.n_suppliers

        next_state, event, i, probs = sourcingEnv.step(random_action)
        sum_check =  np.sum(probs)
        history.append([next_state, event, i, probs])
    
    print(sum_check)