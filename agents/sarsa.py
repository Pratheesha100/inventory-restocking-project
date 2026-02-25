# =============================================================================
# agents/sarsa.py
#
# SARSA Agent
# Author: Member 3
#
# TO DO (Member 3):
# -----------------
# 1. Implement the SARSAAgent class
# 2. Implement the Q-table using a numpy array of shape (n_states, n_actions)
# 3. Implement the train() method using the SARSA update rule:
#       Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
#       NOTE: Unlike Q-Learning, SARSA uses the ACTUAL next action (a')
#             not the greedy max action. This makes SARSA on-policy.
# 4. Implement the choose_action() method using epsilon-greedy strategy
# 5. Implement the evaluate() method to test the learned policy
# 6. Save training rewards to results/ for plotting
# 7. Compare SARSA results against Q-Learning results for the report
#
# Key difference from Q-Learning:
# --------------------------------
# Q-Learning:  Q(s,a) += alpha * [r + gamma * MAX(Q(s',a')) - Q(s,a)]
# SARSA:       Q(s,a) += alpha * [r + gamma * Q(s', a_next) - Q(s,a)]
# where a_next is the action actually chosen by the policy, not the best possible.
#
# Import hyperparameters from config.py — do NOT hardcode them here:
# import config
# alpha   = config.SARSA_ALPHA
# gamma   = config.SARSA_GAMMA
# epsilon = config.SARSA_EPSILON_START
# =============================================================================

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SARSAAgent:
    """
    SARSA Agent — to be implemented by Member 3.
    """

    def __init__(self, n_states, n_actions):
        # TODO: Member 3 — initialise Q-table, hyperparameters
        raise NotImplementedError("Member 3: Please implement SARSAAgent")

    def choose_action(self, state_index):
        # TODO: Member 3 — epsilon-greedy action selection
        raise NotImplementedError("Member 3: Please implement choose_action()")

    def train(self, env, n_episodes):
        # TODO: Member 3 — training loop with SARSA update rule
        raise NotImplementedError("Member 3: Please implement train()")

    def evaluate(self, env, n_episodes):
        # TODO: Member 3 — evaluation loop using greedy policy
        raise NotImplementedError("Member 3: Please implement evaluate()")