from env.SourcingEnvMCTS import *
from opt.mcts import *

if __name__=="__main__":

    sourcingEnv = SourcingEnv(
        lambda_arrival = 599, # or 10
        procurement_cost_vec = np.array([3, 1]),
        supplier_lead_times_vec = np.array([0.5, 0.75]),
        on_times = np.array([3, 1]), 
        off_times = np.array([0.3, 1]))

    initialState = SourcingEnvMCTSWrapper(sourcingEnv)
    searcher = mcts()
    action = searcher.search(initialState=initialState)

    print(action)