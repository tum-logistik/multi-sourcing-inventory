from environment.SourcingEnv import *
import numpy as np

if __name__ == '__main__':
    print("run")

    sourcingEnv = SourcingEnv()

    for i in range(20):
        probs = sourcingEnv.step([i, i+1])
        sum_check =  np.sum(probs)