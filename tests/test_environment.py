# =============================================================================
# tests/test_environment.py
#
# Unit tests for the Inventory RL Environment
# Author: Member 1- Pratheesha
#
# Run with: python tests/test_environment.py
# =============================================================================

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.inventory_env import InventoryEnvironment
import config


def test_reset():
    """Test that reset() returns a valid initial state."""
    env = InventoryEnvironment()
    state = env.reset()

    assert isinstance(state, tuple), "State should be a tuple"
    assert len(state) == 3, "State should have 3 components"

    stock_bin, day_of_week, promo_flag = state
    assert stock_bin in [0, 1, 2], "Stock bin should be 0, 1, or 2"
    assert 0 <= day_of_week <= 6, "Day of week should be between 0 and 6"
    assert promo_flag in [0, 1], "Promo flag should be 0 or 1"

    print("[PASS] test_reset")


def test_step_returns_correct_types():
    """Test that step() returns the correct types."""
    env = InventoryEnvironment()
    env.reset()

    next_state, reward, done, info = env.step(1)  # action 1 = order 10

    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(done, bool), "Done should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"

    print("[PASS] test_step_returns_correct_types")


def test_stock_never_negative():
    """Test that stock level never goes below zero."""
    env = InventoryEnvironment()
    env.reset()
    done = False

    while not done:
        action = 0  # never order anything (worst case)
        _, _, done, info = env.step(action)
        assert info['stock_after'] >= 0, \
            f"Stock went negative: {info['stock_after']}"

    print("[PASS] test_stock_never_negative")


def test_stock_never_exceeds_max():
    """Test that stock never exceeds the maximum capacity."""
    env = InventoryEnvironment()
    env.reset()
    done = False

    while not done:
        action = 2  # always order maximum (20 units)
        _, _, done, info = env.step(action)
        assert info['stock_after'] <= config.MAX_STOCK_CAPACITY, \
            f"Stock exceeded max: {info['stock_after']}"

    print("[PASS] test_stock_never_exceeds_max")


def test_state_to_index_valid_range():
    """Test that state_to_index returns values within the valid range."""
    env = InventoryEnvironment()

    # Test all possible state combinations
    for stock_bin in [0, 1, 2]:
        for day in range(7):
            for promo in [0, 1]:
                state = (stock_bin, day, promo)
                idx = env.state_to_index(state)
                assert 0 <= idx < env.n_states, \
                    f"State index {idx} out of range for state {state}"

    print("[PASS] test_state_to_index_valid_range")


def test_reward_positive_when_selling():
    """Test that reward is positive when selling well with no stockouts."""
    env = InventoryEnvironment()

    # Manually test the reward calculation
    # 10 units sold, 0 leftover, 0 unmet demand
    reward = env._calculate_reward(
        units_sold=10,
        leftover_stock=0,
        unmet_demand=0
    )
    expected = 10 * config.PROFIT_PER_UNIT  # = 50.0
    assert reward == expected, f"Expected {expected}, got {reward}"

    print("[PASS] test_reward_positive_when_selling")


def test_reward_penalised_for_stockout():
    """Test that reward decreases when there is unmet demand."""
    env = InventoryEnvironment()

    reward_no_stockout = env._calculate_reward(10, 0, 0)
    reward_with_stockout = env._calculate_reward(10, 0, 5)

    assert reward_with_stockout < reward_no_stockout, \
        "Reward should be lower with stockouts"

    print("[PASS] test_reward_penalised_for_stockout")


def test_full_episode_runs_without_error():
    """Test that a full episode completes without errors."""
    env = InventoryEnvironment()
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = np.random.randint(0, env.n_actions)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    assert steps > 0, "Episode should have at least one step"
    print(f"[PASS] test_full_episode_runs_without_error "
          f"(steps={steps}, total_reward={total_reward:.2f})")


def test_get_state_space_info():
    """Test that state space info returns correct values."""
    env = InventoryEnvironment()
    info = env.get_state_space_info()

    assert info['n_states'] == 42, f"Expected 42 states, got {info['n_states']}"
    assert info['n_actions'] == 3, f"Expected 3 actions, got {info['n_actions']}"
    assert info['action_space'] == [0, 10, 20]

    print("[PASS] test_get_state_space_info")


def test_reward_penalised_for_overstocking():
    """Test that reward decreases due to storage costs when keeping excess stock."""
    env = InventoryEnvironment()
    
    # Sell 10, leave 0 in storage, 0 unmet
    reward_perfect_stock = env._calculate_reward(units_sold=10, leftover_stock=0, unmet_demand=0)
    
    # Sell 10, but leave 50 units sitting in the back room
    reward_overstocked = env._calculate_reward(units_sold=10, leftover_stock=50, unmet_demand=0)
    
    assert reward_overstocked < reward_perfect_stock, \
        "Reward should be lower when storage costs are incurred"
        
    print("[PASS] test_reward_penalised_for_overstocking")


def test_order_action_increases_stock():
    """Test that taking an order action actually increases the stock before sales."""
    env = InventoryEnvironment()
    env.reset()
    
    initial_stock = env.current_stock # Should be 50
    
    # Look at the demand for day 0 so we can reverse-engineer the math
    demand_day_0 = int(env.data.iloc[0]['Units Sold'])
    
    # Take Action 2 (Order 20 units)
    _, _, _, info = env.step(2) 
    
    # Expected stock: Initial (50) + Ordered (20) - Sold (capped at demand)
    units_sold = min(70, demand_day_0)
    expected_stock_after = 70 - units_sold
    
    assert info['stock_after'] == expected_stock_after, \
        f"Order logic failed. Expected {expected_stock_after}, got {info['stock_after']}"
        
    print("[PASS] test_order_action_increases_stock")


# =============================================================================
# Run all tests
# =============================================================================
if __name__ == '__main__':
    print("=" * 50)
    print("Running Environment Unit Tests")
    print("=" * 50)

    test_reset()
    test_step_returns_correct_types()
    test_stock_never_negative()
    test_stock_never_exceeds_max()
    test_state_to_index_valid_range()
    test_reward_positive_when_selling()
    test_reward_penalised_for_stockout()
    test_full_episode_runs_without_error()
    test_get_state_space_info()
    test_reward_penalised_for_overstocking()
    test_order_action_increases_stock()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)