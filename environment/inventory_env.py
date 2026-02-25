# =============================================================================
# environment/inventory_env.py
#
# Reinforcement Learning Environment for Inventory Restocking
# Author: Member 1
#
# This class simulates a retail store inventory system.
# An RL agent interacts with this environment by choosing how many units
# to order each day, and receives a reward based on profit, storage costs,
# and stockout penalties.
#
# Compatible with Q-Learning (agents/q_learning.py)
# Compatible with SARSA     (agents/sarsa.py)
# =============================================================================

import numpy as np
import pandas as pd
import sys
import os

# Import shared configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class InventoryEnvironment:
    """
    A simulated retail inventory environment for Reinforcement Learning.

    MDP Definition:
    ---------------
    State  : (stock_level_bin, day_of_week, promo_flag)
               - stock_level_bin : 0=Low, 1=Medium, 2=High
               - day_of_week     : 0=Monday ... 6=Sunday
               - promo_flag      : 0=No promotion, 1=Promotion active

    Actions: 0 -> Order 0 units
             1 -> Order 10 units
             2 -> Order 20 units

    Reward : (units_sold × profit_per_unit)
           - (leftover_stock × storage_cost_per_unit)
           - (unmet_demand × shortage_penalty_per_unit)

    Episode: One full pass through the dataset (one year of daily records)
    """

    def __init__(self, data_path=None):
        """
        Initialise the inventory environment.

        Parameters:
        -----------
        data_path : str, optional
            Path to the preprocessed CSV file.
            Defaults to the path set in config.py
        """
        # Use default path from config if not provided
        if data_path is None:
            data_path = config.DATA_PROCESSED_PATH

        # Load the preprocessed dataset
        self.data = pd.read_csv(data_path)
        self.data.reset_index(drop=True, inplace=True)

        # Set random seed for reproducibility
        np.random.seed(config.RANDOM_SEED)

        # -------------------------------------------------------
        # Action Space
        # -------------------------------------------------------
        self.action_space = config.ACTION_SPACE      # [0, 10, 20]
        self.n_actions = len(self.action_space)      # 3

        # -------------------------------------------------------
        # State Space Dimensions
        # -------------------------------------------------------
        self.n_stock_bins = config.STOCK_BINS        # 3
        self.n_days = 7                              # Monday–Sunday
        self.n_promo = 2                             # 0 or 1

        # Total unique states = 3 × 7 × 2 = 42
        self.n_states = self.n_stock_bins * self.n_days * self.n_promo

        # -------------------------------------------------------
        # Environment Limits
        # -------------------------------------------------------
        self.max_stock = config.MAX_STOCK_CAPACITY   # 100 units max
        self.initial_stock = config.INITIAL_STOCK    # Start with 50 units

        # -------------------------------------------------------
        # Reward Function Parameters
        # -------------------------------------------------------
        self.profit_per_unit = config.PROFIT_PER_UNIT               # $5.0
        self.storage_cost_per_unit = config.STORAGE_COST_PER_UNIT   # $1.0
        self.shortage_penalty = config.SHORTAGE_PENALTY_PER_UNIT    # $3.0

        # -------------------------------------------------------
        # Internal State Tracking
        # -------------------------------------------------------
        self.current_stock = self.initial_stock
        self.current_step = 0
        self.max_steps = len(self.data)
        self.done = False

        print(f"[Environment] Loaded {self.max_steps} days of data.")
        print(f"[Environment] Total unique states : {self.n_states}")
        print(f"[Environment] Total actions        : {self.n_actions}")
        print(f"[Environment] Action space         : {self.action_space}")

    # =========================================================================
    # PUBLIC METHODS (used by Q-Learning and SARSA agents)
    # =========================================================================

    def reset(self):
        """
        Reset the environment to its initial state.
        Must be called at the beginning of every new episode.

        Returns:
        --------
        state : tuple
            Initial state as (stock_bin, day_of_week, promo_flag)
        """
        self.current_stock = self.initial_stock
        self.current_step = 0
        self.done = False

        # Get the state for the very first day
        row = self.data.iloc[self.current_step]
        state = self._get_state(self.current_stock, row)

        return state

    def step(self, action_index):
        """
        Execute one timestep (one day) in the environment.

        Parameters:
        -----------
        action_index : int
            Index into the action space (0, 1, or 2)
            0 → Order 0 units
            1 → Order 10 units
            2 → Order 20 units

        Returns:
        --------
        next_state : tuple or None
            Next state (stock_bin, day_of_week, promo_flag).
            None if the episode has ended.
        reward : float
            Reward for this timestep.
        done : bool
            True if the episode has ended.
        info : dict
            Diagnostic information for debugging and analysis.
        """
        if self.done:
            raise RuntimeError(
                "Episode is already finished. Call reset() before stepping again."
            )

        # -------------------------------------------------------
        # Step 1: Convert action index to order quantity
        # -------------------------------------------------------
        order_qty = self.action_space[action_index]

        # -------------------------------------------------------
        # Step 2: Add ordered stock (capped at max capacity)
        # -------------------------------------------------------
        self.current_stock = min(
            self.current_stock + order_qty,
            self.max_stock
        )

        # -------------------------------------------------------
        # Step 3: Get today's demand from the dataset
        # -------------------------------------------------------
        row = self.data.iloc[self.current_step]
        demand = int(row['Units Sold'])

        # -------------------------------------------------------
        # Step 4: Calculate sales and unmet demand
        # -------------------------------------------------------
        # Can't sell more than what's in stock
        units_sold = min(self.current_stock, demand)

        # Demand that couldn't be fulfilled (stockout)
        unmet_demand = max(0, demand - self.current_stock)

        # -------------------------------------------------------
        # Step 5: Update stock after today's sales
        # -------------------------------------------------------
        self.current_stock = self.current_stock - units_sold

        # -------------------------------------------------------
        # Step 6: Calculate today's reward
        # -------------------------------------------------------
        reward = self._calculate_reward(
            units_sold,
            self.current_stock,
            unmet_demand
        )

        # -------------------------------------------------------
        # Step 7: Advance timestep and check if episode is over
        # -------------------------------------------------------
        self.current_step += 1
        self.done = self.current_step >= self.max_steps - 1

        # -------------------------------------------------------
        # Step 8: Get next state (or None if episode ended)
        # -------------------------------------------------------
        if not self.done:
            next_row = self.data.iloc[self.current_step]
            next_state = self._get_state(self.current_stock, next_row)
        else:
            next_state = None

        # -------------------------------------------------------
        # Step 9: Build info dictionary for diagnostics
        # -------------------------------------------------------
        info = {
            'day': self.current_step,
            'order_qty': order_qty,
            'demand': demand,
            'units_sold': units_sold,
            'unmet_demand': unmet_demand,
            'stock_after': self.current_stock,
            'reward_breakdown': {
                'profit': units_sold * self.profit_per_unit,
                'storage_cost': self.current_stock * self.storage_cost_per_unit,
                'shortage_cost': unmet_demand * self.shortage_penalty
            }
        }

        return next_state, reward, self.done, info

    def state_to_index(self, state):
        """
        Convert a state tuple into a single integer index for the Q-table.

        The Q-table in both Q-Learning and SARSA is indexed by a single integer,
        not a tuple. This function provides that conversion.

        Encoding formula:
        index = (stock_bin × 7 × 2) + (day_of_week × 2) + promo_flag

        Parameters:
        -----------
        state : tuple
            State as (stock_bin, day_of_week, promo_flag)

        Returns:
        --------
        index : int
            Integer index in range [0, n_states - 1]

        Example:
        --------
        >>> env.state_to_index((1, 3, 0))
        >>> 20
        """
        if state is None:
            return None

        stock_bin, day_of_week, promo_flag = state

        index = (
            (stock_bin * self.n_days * self.n_promo) +
            (day_of_week * self.n_promo) +
            promo_flag
        )

        return index

    def get_state_space_info(self):
        """
        Returns a summary of the state and action space.
        Useful for initialising the Q-table in agent files.

        Returns:
        --------
        dict with n_states and n_actions
        """
        return {
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'action_space': self.action_space
        }

    # =========================================================================
    # PRIVATE METHODS (internal helpers, not called by agents directly)
    # =========================================================================

    def _get_state(self, stock_level, row):
        """
        Convert raw stock level and data row into a discrete state tuple.

        Parameters:
        -----------
        stock_level : int or float
            Current stock level
        row : pd.Series
            One row from the dataset containing Day_of_Week and Promo_Flag

        Returns:
        --------
        state : tuple of (stock_bin, day_of_week, promo_flag)
        """
        # Discretise stock level
        if stock_level < config.STOCK_LOW_THRESHOLD:
            stock_bin = 0   # Low stock
        elif stock_level < config.STOCK_HIGH_THRESHOLD:
            stock_bin = 1   # Medium stock
        else:
            stock_bin = 2   # High stock

        day_of_week = int(row['Day_of_Week'])   # 0 = Monday, 6 = Sunday
        promo_flag = int(row['Promo_Flag'])      # 0 = no promo, 1 = promo

        return (stock_bin, day_of_week, promo_flag)

    def _calculate_reward(self, units_sold, leftover_stock, unmet_demand):
        """
        Calculate the reward signal for the current timestep.

        Formula:
        --------
        Reward = (units_sold × profit_per_unit)
               - (leftover_stock × storage_cost_per_unit)
               - (unmet_demand × shortage_penalty_per_unit)

        The agent is rewarded for selling units, penalised for excess
        inventory, and penalised more heavily for failing to meet demand.

        Parameters:
        -----------
        units_sold     : int   - units successfully sold today
        leftover_stock : int   - unsold units remaining after today
        unmet_demand   : int   - demand that could not be fulfilled

        Returns:
        --------
        reward : float
        """
        profit = units_sold * self.profit_per_unit
        storage_cost = leftover_stock * self.storage_cost_per_unit
        shortage_cost = unmet_demand * self.shortage_penalty

        reward = profit - storage_cost - shortage_cost

        return round(reward, 2)