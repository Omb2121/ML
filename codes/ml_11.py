
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure UTF-8 output when supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("\nLibraries imported successfully!")


# ---------------------------------------------------------
# 1. OPTIONAL DATASET LOADING
# ---------------------------------------------------------
# The Q-Learning algorithm does not require this CSV file.
try:
    df = pd.read_csv("grid_world_dataset.csv")

    print("\nGrid World Dataset:")
    print(df)
    print("Dataset shape:", df.shape)

except FileNotFoundError:
    print(
        "\nOptional file 'grid_world_dataset.csv' not found; "
        "continuing without it."
    )

# ---------------------------------------------------------
# 2. CREATE THE GRID WORLD ENVIRONMENT
# ---------------------------------------------------------
GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4

START_STATE = 0
GOAL_STATE = 15

# Actions: Up, Down, Left, Right
ACTIONS = ["Up", "Down", "Left", "Right"]
ARROWS = ["↑", "↓", "←", "→"]


def take_action(state, action):
    """Execute an action and return the next state and reward."""

    row, col = divmod(state, GRID_SIZE)

    if action == 0:       # Up
        row = max(0, row - 1)

    elif action == 1:     # Down
        row = min(GRID_SIZE - 1, row + 1)

    elif action == 2:     # Left
        col = max(0, col - 1)

    elif action == 3:     # Right
        col = min(GRID_SIZE - 1, col + 1)

    next_state = row * GRID_SIZE + col

    # Reward for reaching the goal; penalty otherwise
    reward = 10 if next_state == GOAL_STATE else -1

    return next_state, reward


print("\nGrid World environment created successfully!")
print("Start state:", START_STATE)
print("Goal state:", GOAL_STATE)
print("Available actions:", ACTIONS)

# ---------------------------------------------------------
# 3. SET Q-LEARNING HYPERPARAMETERS
# ---------------------------------------------------------
ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor
EPISODES = 1000
MAX_STEPS = 100

# ---------------------------------------------------------
# 4. DEFINE THE Q-LEARNING ALGORITHM
# ---------------------------------------------------------
def train_q_learning(epsilon, seed=42):
    """Train Q-Learning using epsilon-greedy action selection."""

    rng = np.random.default_rng(seed)

    Q = np.zeros((N_STATES, N_ACTIONS))

    episode_rewards = []
    episode_steps = []
    episode_success = []

    for episode in range(EPISODES):
        state = START_STATE
        total_reward = 0
        success = False

        for step in range(MAX_STEPS):

            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = rng.integers(N_ACTIONS)

            else:
                best_actions = np.flatnonzero(
                    Q[state] == np.max(Q[state])
                )

                action = rng.choice(best_actions)

            # Take action in the environment
            next_state, reward = take_action(state, action)

            # Q-Learning update
            if next_state == GOAL_STATE:
                target = reward

            else:
                target = (
                    reward
                    + GAMMA * np.max(Q[next_state])
                )

            Q[state, action] += ALPHA * (
                target - Q[state, action]
            )

            total_reward += reward
            state = next_state

            # Stop when the goal is reached
            if state == GOAL_STATE:
                success = True
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        episode_success.append(int(success))

    return (
        Q,
        np.array(episode_rewards),
        np.array(episode_steps),
        np.array(episode_success)
    )


print("\nQ-Learning algorithm defined successfully!")

# ---------------------------------------------------------
# 5. TRAIN USING DIFFERENT EPSILON VALUES
# ---------------------------------------------------------
epsilon_values = [0.5, 0.1, 0.01]

results = {}

for epsilon in epsilon_values:
    Q, rewards, steps, success = train_q_learning(
        epsilon=epsilon,
        seed=42
    )

    results[epsilon] = {
        "Q": Q,
        "rewards": rewards,
        "steps": steps,
        "success": success
    }

print("Training completed for all epsilon values!")

# ---------------------------------------------------------
# 6. COMPARE PERFORMANCE OVER THE LAST 100 EPISODES
# ---------------------------------------------------------
summary = []

for epsilon, result in results.items():
    last_100 = slice(-100, None)

    summary.append({
        "Epsilon": epsilon,
        "Average Reward": round(
            result["rewards"][last_100].mean(), 2
        ),
        "Success Rate (%)": round(
            result["success"][last_100].mean() * 100, 2
        ),
        "Average Steps": round(
            result["steps"][last_100].mean(), 2
        )
    })

summary_df = pd.DataFrame(summary)

print("\nPerformance Comparison:")
print(summary_df.to_string(index=False))

# ---------------------------------------------------------
# 7. PLOT AVERAGE REWARD DURING TRAINING
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

for epsilon, result in results.items():
    moving_avg = pd.Series(
        result["rewards"]
    ).rolling(
        window=50,
        min_periods=1
    ).mean()

    plt.plot(
        moving_avg,
        label=f"Epsilon = {epsilon}"
    )

plt.title("Q-Learning: Average Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Average Reward (50-episode window)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. PLOT SUCCESS RATE DURING TRAINING
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

for epsilon, result in results.items():
    success_rate = pd.Series(
        result["success"]
    ).rolling(
        window=50,
        min_periods=1
    ).mean() * 100

    plt.plot(
        success_rate,
        label=f"Epsilon = {epsilon}"
    )

plt.title("Q-Learning: Success Rate During Training")
plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. DISPLAY THE FINAL Q-TABLE FOR EPSILON = 0.1
# ---------------------------------------------------------
Q_final = results[0.1]["Q"]

q_table = pd.DataFrame(
    Q_final,
    columns=ACTIONS,
    index=[f"State {i}" for i in range(N_STATES)]
)

print("\nFinal Q-Table (Epsilon = 0.1):")
print(q_table.round(2).to_string())

# ---------------------------------------------------------
# 10. DISPLAY THE LEARNED POLICY
# ---------------------------------------------------------
policy = []

for state in range(N_STATES):
    if state == START_STATE:
        policy.append("S")

    elif state == GOAL_STATE:
        policy.append("G")

    else:
        best_action = np.argmax(Q_final[state])
        policy.append(ARROWS[best_action])

policy_grid = np.array(policy).reshape(
    GRID_SIZE,
    GRID_SIZE
)

print("\nLearned Optimal Policy:")
print(policy_grid)

# Plot the learned policy
plt.figure(figsize=(6, 6))

plt.imshow(
    np.zeros((GRID_SIZE, GRID_SIZE)),
    cmap="Blues"
)

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        plt.text(
            col,
            row,
            policy_grid[row, col],
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold"
        )

plt.xticks(range(GRID_SIZE))
plt.yticks(range(GRID_SIZE))
plt.title("Learned Policy (S = Start, G = Goal)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 11. EXECUTE THE LEARNED POLICY
# ---------------------------------------------------------
state = START_STATE
path = [state]
total_reward = 0

for step in range(50):
    action = np.argmax(Q_final[state])

    next_state, reward = take_action(state, action)

    total_reward += reward
    state = next_state
    path.append(state)

    if state == GOAL_STATE:
        break

print("\nPath followed:", path)
print("Number of moves:", len(path) - 1)
print("Total reward:", total_reward)

if state == GOAL_STATE:
    print("Result: Goal reached successfully!")
else:
    print("Result: Goal not reached within the step limit.")

# ---------------------------------------------------------
# 12. DISPLAY THE FINAL RESULT
# ---------------------------------------------------------
best = summary_df.loc[
    summary_df["Success Rate (%)"].idxmax()
]

print("\nRESULT")
print("-" * 40)

print(
    "Best epsilon by observed success rate:",
    best["Epsilon"]
)

print("Success rate:", best["Success Rate (%)"], "%")
print("Average reward:", best["Average Reward"])
print("Average steps:", best["Average Steps"])
