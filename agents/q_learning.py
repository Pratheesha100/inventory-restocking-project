# =============================================================================
# agents/q_learning.py
# Q-Learning Agent

# Author : Aweesha Wijesundara | W.M.A.T Wijesundara | IT22183668
# =============================================================================

import numpy as np
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class QLearningAgent:
    def __init__(self, n_states, n_actions):
        # State & action sizes
        self.n_states = n_states
        self.n_actions = n_actions

        # Q-table
        self.q_table = np.zeros((n_states, n_actions))

        # Hyperparameters (from config)
        self.alpha = config.QL_ALPHA
        self.gamma = config.QL_GAMMA
        self.epsilon = config.QL_EPSILON_START
        self.epsilon_min = config.QL_EPSILON_END
        self.epsilon_decay = config.QL_EPSILON_DECAY

        # Tracking
        self.training_rewards = []

    def choose_action(self, state_index):
        """
        Epsilon-greedy policy
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return np.argmax(self.q_table[state_index])

    def train(self, env, n_episodes):
        """
        Train using Q-Learning
        """
        for episode in range(n_episodes):
            state = env.reset()
            state_idx = env.state_to_index(state)

            done = False
            total_reward = 0

            while not done:
                # Select action
                action = self.choose_action(state_idx)

                # Step environment
                next_state, reward, done, info = env.step(action)
                next_state_idx = env.state_to_index(next_state)

                # Q-Learning update
                current_q = self.q_table[state_idx, action]
                max_future_q = np.max(self.q_table[next_state_idx])

                self.q_table[state_idx, action] = current_q + self.alpha * (
                    reward + self.gamma * max_future_q - current_q
                )

                # Move to next state
                state_idx = next_state_idx
                total_reward += reward

            # Store reward
            self.training_rewards.append(total_reward)

            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            # Progress log
            if (episode + 1) % 100 == 0:
                print(
                    f"[Q-Learning] Episode {episode+1}/{n_episodes} | "
                    f"Reward: {total_reward:.2f} | Epsilon: {self.epsilon:.4f}"
                )

        # Save results
        self._save_results()

        # IMPORTANT: return rewards
        return self.training_rewards

    def evaluate(self, env, n_episodes):
        """
        Evaluate greedy policy
        """
        total_reward = 0
        stockouts = 0
        overstocks = 0

        for _ in range(n_episodes):
            state = env.reset()
            state_idx = env.state_to_index(state)

            done = False

            while not done:
                action = np.argmax(self.q_table[state_idx])

                next_state, reward, done, info = env.step(action)
                next_state_idx = env.state_to_index(next_state)

                total_reward += reward

                stockouts += info.get("stockout", 0)
                overstocks += info.get("overstock", 0)

                state_idx = next_state_idx

        return {
            "avg_reward": total_reward / n_episodes,
            "stockout_rate": stockouts / n_episodes,
            "overstock_rate": overstocks / n_episodes
        }

    def _save_results(self):
        """
        Save training rewards
        """
        results_dir = os.path.join("results", "q_learning")
        os.makedirs(results_dir, exist_ok=True)

        np.save(
            os.path.join(results_dir, "training_rewards.npy"),
            np.array(self.training_rewards)
        )

        print(f"[Q-Learning] Results saved to {results_dir}")