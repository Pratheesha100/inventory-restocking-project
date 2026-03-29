import numpy as np
import matplotlib.pyplot as plt
import config
from environment.inventory_env import InventoryEnvironment
from agents.q_learning import QLearningAgent

print("\n--- Testing Q-Learning Agent Independently ---")

# 1. Load Environment
env = InventoryEnvironment()
space_info = env.get_state_space_info()
n_states = space_info["n_states"]
n_actions = space_info["n_actions"]

# 2. Initialize and Train Q-Learning
print("\nTraining Q-Learning Agent...")
q_learning_agent = QLearningAgent(n_states, n_actions)
q_learning_rewards = q_learning_agent.train(env, config.N_EPISODES)
print(
    f"Training complete! Average reward over last 100 episodes: {np.mean(q_learning_rewards[-100:]):.2f}"
)

# 3. Evaluate Q-Learning
print("\nEvaluating Q-Learning Agent (Zero Exploration)...")
q_learning_eval = q_learning_agent.evaluate(env, config.TEST_EPISODES)
print(f"Average Reward per Episode: {q_learning_eval['avg_reward']:.2f}")
print(f"Stockout Rate: {q_learning_eval['stockout_rate']:.2f}%")
print(f"Overstock Rate: {q_learning_eval['overstock_rate']:.2f}%")

# 4. Quick Plot to verify learning
plt.plot(
    np.convolve(q_learning_rewards, np.ones(100) / 100, mode="valid"), color="blue"
)
plt.title("Q-Learning: Smoothed Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

print("\nIf you see this and the graph pops up, your Q-Learning code works perfectly!")
