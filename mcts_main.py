from env.SourcingEnvMCTS import *
from opt.mcts import *

if __name__=="__main__":
    initialState = SourcingEnvMCTSWrapper()
    searcher = mcts(timeLimit=1000)
    action = searcher.search(initialState=initialState)

    print(action)