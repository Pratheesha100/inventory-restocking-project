# =============================================================================
# agents/q_learning.py
#
# Q-Learning Agent
# Author: Member 2
#
# TO DO (Member 2):
# -----------------
# 1. Implement the QLearningAgent class
# 2. Implement the Q-table using a numpy array of shape (n_states, n_actions)
# 3. Implement the train() method using the Q-Learning update rule:
#       Q(s,a) <- Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
# 4. Implement the choose_action() method using epsilon-greedy strategy
# 5. Implement the evaluate() method to test the learned policy
# 6. Save training rewards to results/ for plotting
#
# How to use the environment:
# ---------------------------
# from environment import InventoryEnvironment
# env = InventoryEnvironment()
# state = env.reset()
# state_idx = env.state_to_index(state)
# next_state, reward, done, info = env.step(action_index)
#
# Get state/action space sizes:
# info = env.get_state_space_info()
# n_states  = info['n_states']    # 42
# n_actions = info['n_actions']   # 3
#
# Import hyperparameters from config.py — do NOT hardcode them here:
# import config
# alpha   = config.QL_ALPHA
# gamma   = config.QL_GAMMA
# epsilon = config.QL_EPSILON_START
# =============================================================================

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class QLearningAgent:
    """
    Q-Learning Agent — to be implemented by Member 2.
    """

    def __init__(self, n_states, n_actions):
        # TODO: Member 2 — initialise Q-table, hyperparameters
        raise NotImplementedError("Member 2: Please implement QLearningAgent")

    def choose_action(self, state_index):
        # TODO: Member 2 — epsilon-greedy action selection
        raise NotImplementedError("Member 2: Please implement choose_action()")

    def train(self, env, n_episodes):
        # TODO: Member 2 — training loop with Q-Learning update rule
        raise NotImplementedError("Member 2: Please implement train()")

    def evaluate(self, env, n_episodes):
        # TODO: Member 2 — evaluation loop using greedy policy
        raise NotImplementedError("Member 2: Please implement evaluate()")
from .sarsa import SARSAAgent