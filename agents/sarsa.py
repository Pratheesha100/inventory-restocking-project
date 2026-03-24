# =============================================================================
# SARSA (State-Action-Reward-State-Action) Agent Implementation
# Author: Dinitha Fernando
#
# This agent uses an on-policy Temporal Difference control algorithm to learn
# the optimal inventory restocking policy.
# =============================================================================

import numpy as np
import sys
import os

# Import shared configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SarsaAgent:
    """
    SARSA Agent for the Inventory Optimization Problem.
    """

    def __init__(self, n_states, n_actions):
        """
        Initialises the SARSA agent with hyperparameters from config.py
        and an empty Q-table.
        """
        self.n_states = n_states
        self.n_actions = n_actions

        # Load Hyperparameters
        self.alpha = config.SARSA_ALPHA  # Learning rate
        self.gamma = config.SARSA_GAMMA  # Discount factor
        self.epsilon = config.SARSA_EPSILON_START  # Exploration rate
        self.epsilon_min = config.SARSA_EPSILON_END
        self.epsilon_decay = config.SARSA_EPSILON_DECAY

        # Initialise Q-table with zeros
        # Rows = states (42), Columns = actions (3)
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def choose_action(self, state_index):
        """
        Selects an action using the epsilon-greedy policy.

        Parameters:
        -----------
        state_index : int
            The integer index representing the current state.

        Returns:
        --------
        action_index : int
            The index of the chosen action (0, 1, or 2).
        """
        # Explore: Choose a random action
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Exploit: Choose the action with the highest Q-value for the current state
        else:
            return np.argmax(self.q_table[state_index])

    def update(self, state_idx, action_idx, reward, next_state_idx, next_action_idx):
        """
        Updates the Q-table using the SARSA update rule.
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]

        Notice that unlike Q-Learning, we use the actual next_action_idx
        that the agent will take, not the theoretical max action.
        """
        # If next_state is None (episode ended), target is just the reward
        if next_state_idx is None:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state_idx, next_action_idx]

        # Current Q-value prediction
        predict = self.q_table[state_idx, action_idx]

        # Update the Q-value in the table
        self.q_table[state_idx, action_idx] += self.alpha * (target - predict)

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon) to shift the agent
        from exploration to exploitation over time.
        Call this at the end of every episode.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
