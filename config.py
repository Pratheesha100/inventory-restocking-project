# =============================================================================
# config.py
# Shared configuration file for the Inventory RL Project
# All hyperparameters and paths are defined here.
# Members 2 and 3 should import from this file rather than hardcoding values.
# =============================================================================

import os

# =============================================================================
# FILE PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "retail_inventory.csv")
DATA_PROCESSED_PATH = os.path.join(
    BASE_DIR, "data", "processed", "preprocessed_product.csv"
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================
MAX_STOCK_CAPACITY = 100  # Maximum units the store can hold
INITIAL_STOCK = 50  # Starting stock at beginning of each episode

# State space
STOCK_BINS = 3  # Low(0), Medium(1), High(2)
# Thresholds: Low = 0-19, Medium = 20-59, High = 60-100
STOCK_LOW_THRESHOLD = 20
STOCK_HIGH_THRESHOLD = 60

# Action space: how many units to order
ACTION_SPACE = [0, 10, 20]  # Order nothing, order 10, order 20

# Reward function parameters
PROFIT_PER_UNIT = 5.0  # Profit earned per unit sold ($)
STORAGE_COST_PER_UNIT = 1.0  # Cost per unsold unit remaining per day ($)
SHORTAGE_PENALTY_PER_UNIT = 3.0  # Penalty per unit of unmet demand ($)

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
N_EPISODES = 5000  # Number of training episodes
MAX_STEPS_PER_EPISODE = 365  # Max steps per episode (1 year of days)
TEST_EPISODES = 100  # Number of episodes for evaluation

# =============================================================================
# Q-LEARNING HYPERPARAMETERS (Member 2)
# =============================================================================
QL_ALPHA = 0.1  # Learning rate
QL_GAMMA = 0.99  # Discount factor
QL_EPSILON_START = 1.0  # Starting exploration rate
QL_EPSILON_END = 0.01  # Minimum exploration rate
QL_EPSILON_DECAY = 0.995  # Decay rate per episode

# =============================================================================
# SARSA HYPERPARAMETERS (Member 3)
# =============================================================================
SARSA_ALPHA = 0.1  # Learning rate (same as Q-Learning for fair comparison)
SARSA_GAMMA = 0.99  # Discount factor (same as Q-Learning)
SARSA_EPSILON_START = 1.0  # Starting exploration rate
SARSA_EPSILON_END = 0.01  # Minimum exploration rate
SARSA_EPSILON_DECAY = 0.995  # Decay rate per episode

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
RANDOM_SEED = 42
