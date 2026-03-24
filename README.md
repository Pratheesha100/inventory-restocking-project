# Inventory Restocking Optimization using Reinforcement Learning

## Project Overview

This project applies Reinforcement Learning (Q-Learning and SARSA) to solve an
inventory restocking problem. An RL agent learns the optimal daily ordering policy
to maximize profit while minimizing storage costs and stockout penalties.

## Dataset

- **Source:** Retail Store Inventory Forecasting Dataset
- **Link:** https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset
- **Download the dataset manually from Kaggle and place the CSV file inside `data/raw/`**

## Project Structure

```
inventory-rl-project/
│
├── data/
│   ├── raw/                    # Original downloaded dataset (not tracked by git)
│   └── processed/              # Cleaned and preprocessed data (not tracked by git)
│
├── environment/
│   ├── __init__.py
│   └── inventory_env.py        # RL simulation environment (Member 1)
│
├── agents/
│   ├── __init__.py
│   ├── q_learning.py           # Q-Learning agent (Member 2)
│   └── sarsa.py                # SARSA agent (Member 3)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis (Member 1)
│   └── 02_preprocessing.ipynb  # Data Preprocessing (Member 1)
│
├── results/
│   └── plots/                  # Generated charts and graphs
│
├── tests/
│   ├── test_environment.py     # Unit tests for environment (Member 1)
│   ├── test_q_learning.py      # Unit tests for Q-Learning (Member 2)
│   └── test_sarsa.py           # Unit tests for SARSA (Member 3)
│
├── reports/                    # Report documents
│
├── config.py                   # Shared configuration and hyperparameters
├── main.py                     # Main script to run training and evaluation
├── requirements.txt            # Python dependencies
└── README.md
```

## Team Members and Responsibilities

| Members             | Role                        | Responsibilities                                          |
| ------------------- | --------------------------- | --------------------------------------------------------- |
| Pratheesha Silva    | Environment & Data Engineer | Data preprocessing, RL environment, shared infrastructure |
| Aweesha Wijesundara | Q-Learning Specialist       | Q-Learning implementation, tuning, results                |
| Dinitha Fernando    | SARSA Specialist            | SARSA implementation, tuning, comparative analysis        |

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Pratheesha100/inventory-restocking-project.git
cd inventory-restocking-project
```

### 2. Create and activate virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

- Go to the Kaggle link above
- Download the CSV file
- Place it inside `data/raw/` folder

### 5. Run preprocessing (Member 1's notebook)

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### 6. Run training

```bash
python main.py
```

## Algorithms

- **Q-Learning** — Off-policy Temporal Difference learning
- **SARSA** — On-policy Temporal Difference learning

## Results

Results and comparison plots are saved to `results/plots/` after running `main.py`.
