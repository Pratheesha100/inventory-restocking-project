# =============================================================================
# main.py
#
# Main entry point for the Inventory RL Project
# Trains both Q-Learning and SARSA agents and compares their results
#
# Run with: python main.py
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
from environment.inventory_env import InventoryEnvironment
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


def plot_learning_curves(ql_rewards, sarsa_rewards):
    """
    Plot and save learning curves for both agents side by side.

    Parameters:
    -----------
    ql_rewards    : list of total rewards per episode (Q-Learning)
    sarsa_rewards : list of total rewards per episode (SARSA)
    """
    # Smooth rewards using a rolling average for readability
    window = 100
    ql_smooth = np.convolve(ql_rewards, np.ones(window) / window, mode="valid")
    sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(ql_rewards, alpha=0.3, color="blue", label="Raw")
    plt.plot(ql_smooth, color="blue", linewidth=2, label=f"Smoothed ({window}-ep avg)")
    plt.title("Q-Learning: Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sarsa_rewards, alpha=0.3, color="orange", label="Raw")
    plt.plot(
        sarsa_smooth, color="orange", linewidth=2, label=f"Smoothed ({window}-ep avg)"
    )
    plt.title("SARSA: Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "learning_curves.png")
    plt.savefig(path, dpi=150)
    # plt.show() # commented to avoid popping up plot image, Already saving to plots folder
    print(f"[Saved] Learning curves → {path}")


def plot_comparison(ql_rewards, sarsa_rewards):
    """
    Plot both agents' smoothed learning curves on the same graph.
    """
    window = 100
    ql_smooth = np.convolve(ql_rewards, np.ones(window) / window, mode="valid")
    sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(ql_smooth, color="blue", linewidth=2, label="Q-Learning")
    plt.plot(sarsa_smooth, color="orange", linewidth=2, label="SARSA")
    plt.title("Q-Learning vs SARSA: Learning Curve Comparison")
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Total Reward ({window}-episode window)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "comparison.png")
    plt.savefig(path, dpi=150)
    # plt.show() # commented to avoid popping up plot image, Already saving to plots folder
    print(f"[Saved] Comparison plot → {path}")


def plot_policy_heatmaps(ql_agent, sarsa_agent):
    """
    Plot and save the learned policies (action with highest Q-value) for both agents side-by-side.
    """
    # Extract the greedy policy (0=0 units, 1=10 units, 2=20 units)
    policy_ql = np.argmax(ql_agent.q_table, axis=1).reshape((3, 14))
    policy_sarsa = np.argmax(sarsa_agent.q_table, axis=1).reshape((3, 14))

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

    x_labels = ["M", "T", "W", "Th", "F", "Sa", "Su"] * 2
    y_labels = ["Low", "Medium", "High"]

    # --- Plot 1: Q-Learning Policy ---
    sns.heatmap(
        policy_ql,
        cmap="Reds",
        annot=True,
        cbar=False,
        xticklabels=x_labels,
        yticklabels=y_labels,
        ax=axes[0],
    )
    axes[0].set_title("Q-Learning Policy (0=None, 1=10, 2=20)")
    axes[0].set_xlabel("Day of Week (Left: No Promo | Right: Promo Active)")
    axes[0].set_ylabel("Stock Level")

    # --- Plot 2: SARSA Policy ---
    sns.heatmap(
        policy_sarsa,
        cmap="Blues",
        annot=True,
        cbar=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
        ax=axes[1],
    )
    axes[1].set_title("SARSA Policy (0=None, 1=10, 2=20)")
    axes[1].set_xlabel("Day of Week (Left: No Promo | Right: Promo Active)")

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "policy_comparison_heatmap.png")
    plt.savefig(path, dpi=150)
    # plt.show() # Commented out so the script doesn't freeze
    print(f"[Saved] Policy comparison heatmap → {path}")


def print_results_table(ql_eval, sarsa_eval):
    """
    Print a formatted comparison table of evaluation results.

    Parameters:
    -----------
    ql_eval    : dict with avg_reward, stockout_rate, overstock_rate
    sarsa_eval : dict with avg_reward, stockout_rate, overstock_rate
    """
    print("\n" + "=" * 55)
    print(f"{'EVALUATION RESULTS':^55}")
    print("=" * 55)
    print(f"{'Metric':<30} {'Q-Learning':>10} {'SARSA':>10}")
    print("-" * 55)
    print(
        f"{'Avg Reward per Episode':<30} {ql_eval['avg_reward']:>10.2f} "
        f"{sarsa_eval['avg_reward']:>10.2f}"
    )
    print(
        f"{'Stockout Rate (%)':<30} {ql_eval['stockout_rate']:>10.2f} "
        f"{sarsa_eval['stockout_rate']:>10.2f}"
    )
    print(
        f"{'Overstock Rate (%)':<30} {ql_eval['overstock_rate']:>10.2f} "
        f"{sarsa_eval['overstock_rate']:>10.2f}"
    )
    print("=" * 55)


if __name__ == "__main__":
    ensure_directories()

    print("\n" + "=" * 55)
    print("  Inventory Restocking RL - Training & Evaluation")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Initialise environment
    # ------------------------------------------------------------------
    env = InventoryEnvironment()
    space_info = env.get_state_space_info()
    n_states = space_info["n_states"]
    n_actions = space_info["n_actions"]

    # ------------------------------------------------------------------
    # Train Q-Learning Agent (Member 2 implements QLearningAgent)
    # ------------------------------------------------------------------
    print("\n[1/4] Training Q-Learning Agent...")
    ql_agent = QLearningAgent(n_states, n_actions)
    ql_rewards = ql_agent.train(env, config.N_EPISODES)
    print(
        f"      Training complete. Final avg reward: "
        f"{np.mean(ql_rewards[-100:]):.2f}"
    )

    # ------------------------------------------------------------------
    # Train SARSA Agent (Member 3 implements SARSAAgent)
    # ------------------------------------------------------------------
    print("\n[2/4] Training SARSA Agent...")
    sarsa_agent = SARSAAgent(n_states, n_actions)
    sarsa_rewards = sarsa_agent.train(env, config.N_EPISODES)
    print(
        f"      Training complete. Final avg reward: "
        f"{np.mean(sarsa_rewards[-100:]):.2f}"
    )

    # ------------------------------------------------------------------
    # Evaluate both agents
    # ------------------------------------------------------------------
    print("\n[3/4] Evaluating agents...")
    ql_eval = ql_agent.evaluate(env, config.TEST_EPISODES)
    sarsa_eval = sarsa_agent.evaluate(env, config.TEST_EPISODES)
    print_results_table(ql_eval, sarsa_eval)

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    print("\n[4/4] Generating plots...")
    print("      ... generating learning curves (this might take a few seconds)")
    plot_learning_curves(ql_rewards, sarsa_rewards)

    print("      ... generating comparison plot")
    plot_comparison(ql_rewards, sarsa_rewards)

    print("      ... generating policy heatmaps")
    plot_policy_heatmaps(ql_agent, sarsa_agent)

    print("\nDone! All results saved to results/plots/")
