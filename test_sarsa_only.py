import numpy as np
import matplotlib.pyplot as plt
import config
from environment.inventory_env import InventoryEnvironment
from agents.sarsa import SARSAAgent

print("\n--- Testing SARSA Agent Independently ---")

# 1. Load Environment
env = InventoryEnvironment()
space_info = env.get_state_space_info()
n_states = space_info["n_states"]
n_actions = space_info["n_actions"]

# 2. Initialize and Train SARSA
print("\nTraining SARSA Agent...")
sarsa_agent = SARSAAgent(n_states, n_actions)
sarsa_rewards = sarsa_agent.train(env, config.N_EPISODES)
print(
    f"Training complete! Average reward over last 100 episodes: {np.mean(sarsa_rewards[-100:]):.2f}"
)

# 3. Evaluate SARSA
print("\nEvaluating SARSA Agent (Zero Exploration)...")
sarsa_eval = sarsa_agent.evaluate(env, config.TEST_EPISODES)
print(f"Average Reward per Episode: {sarsa_eval['avg_reward']:.2f}")
print(f"Stockout Rate: {sarsa_eval['stockout_rate']:.2f}%")
print(f"Overstock Rate: {sarsa_eval['overstock_rate']:.2f}%")

# 4. Quick Plot to verify learning
plt.plot(np.convolve(sarsa_rewards, np.ones(100) / 100, mode="valid"), color="orange")
plt.title("SARSA: Smoothed Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

print("\nIf you see this and the graph pops up, your code works perfectly!")
