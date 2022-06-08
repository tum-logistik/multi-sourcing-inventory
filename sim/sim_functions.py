import numpy as np

# simulation functions

def cost_calc(state, h_cost = 5, b_penalty = 5):
    cost = state.s * h_cost if state.s > 0 else np.abs(state.s * b_penalty)
    return cost



