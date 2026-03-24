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


class SARSAAgent:
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

    def train(self, env, n_episodes):
        """
        Trains the agent by interacting with the environment over many episodes.

        Returns:
        --------
        rewards_history : list
            Total reward accumulated in each episode.
        """
        rewards_history = []

        for episode in range(n_episodes):
            state = env.reset()
            state_idx = env.state_to_index(state)

            # SARSA DIFFERENCE: We must choose the first action BEFORE the loop starts
            action_idx = self.choose_action(state_idx)

            total_reward = 0
            done = False

            while not done:
                # 1. Take the action in the environment
                next_state, reward, done, info = env.step(action_idx)
                next_state_idx = env.state_to_index(next_state)

                # 2. Choose the NEXT action based on the NEXT state (On-policy)
                if not done:
                    next_action_idx = self.choose_action(next_state_idx)
                else:
                    next_action_idx = None  # Doesn't matter if the episode is done

                # 3. Update the Q-table using the SARSA rule
                self.update(
                    state_idx, action_idx, reward, next_state_idx, next_action_idx
                )

                # 4. Transition to the next timestep
                state_idx = next_state_idx
                action_idx = next_action_idx
                total_reward += reward

            # Decay exploration rate at the end of the episode
            self.decay_epsilon()
            rewards_history.append(total_reward)

        return rewards_history

    def evaluate(self, env, n_episodes):
        """
        Evaluates the trained agent by running it with ZERO exploration (pure exploitation).
        Calculates metrics required by main.py for the report.
        """
        # Temporarily turn off exploration
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        total_rewards = []
        total_days = 0
        stockout_days = 0
        overstock_days = 0

        for _ in range(n_episodes):
            state = env.reset()
            state_idx = env.state_to_index(state)
            action_idx = self.choose_action(state_idx)

            episode_reward = 0
            done = False

            while not done:
                next_state, reward, done, info = env.step(action_idx)
                next_state_idx = env.state_to_index(next_state)

                if not done:
                    next_action_idx = self.choose_action(next_state_idx)

                state_idx = next_state_idx
                action_idx = next_action_idx

                episode_reward += reward
                total_days += 1

                # Track metrics for the evaluation table
                if info["unmet_demand"] > 0:
                    stockout_days += 1
                if (
                    info["stock_after"] >= env.max_stock * 0.9
                ):  # >90% capacity is overstock
                    overstock_days += 1

            total_rewards.append(episode_reward)

        # Restore the original exploration rate
        self.epsilon = original_epsilon

        # Return the exact dictionary format main.py expects
        return {
            "avg_reward": np.mean(total_rewards),
            "stockout_rate": (stockout_days / total_days) * 100,
            "overstock_rate": (overstock_days / total_days) * 100,
        }
