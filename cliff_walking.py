"""
File: src/cliff_walking.py
Description: Advanced Reinforcement Learning implementation comparing 
             Q-Learning (Off-policy) and SARSA (On-policy) algorithms 
             on the Cliff Walking environment with Heatmap analytics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CliffWalkingEnvironment:
    def __init__(self, height=4, width=12):
        self.height = height
        self.width = width
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.current_state = self.start_state

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        """
        Actions: 0: Up, 1: Right, 2: Down, 3: Left
        """
        r, c = self.current_state
        if action == 0: r = max(0, r - 1)
        elif action == 1: c = min(self.width - 1, c + 1)
        elif action == 2: r = min(self.height - 1, r + 1)
        elif action == 3: c = max(0, c - 1)

        next_state = (r, c)
        
        # Check if agent falls off the cliff
        if r == 3 and 1 <= c <= 10:
            reward = -100
            next_state = self.start_state
        elif next_state == self.goal_state:
            reward = 0  # Reached goal
        else:
            reward = -1  # Step penalty

        self.current_state = next_state
        done = (next_state == self.goal_state)
        return next_state, reward, done

def select_action(state, q_table, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_table[state[0], state[1]])

def train_q_learning(env, episodes=400):
    q_table = np.zeros((env.height, env.width, 4))
    alpha, gamma = 0.1, 0.99
    epsilon, epsilon_decay, min_epsilon = 1.0, 0.99, 0.01
    rewards_history = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, q_table, epsilon)
            next_state, reward, done = env.step(action)

            # Q-learning (Off-policy): uses max Q for next state
            best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
            td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
            q_table[state[0], state[1], action] += alpha * (td_target - q_table[state[0], state[1], action])

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
    return q_table, rewards_history

def train_sarsa(env, episodes=400):
    q_table = np.zeros((env.height, env.width, 4))
    alpha, gamma = 0.1, 0.99
    epsilon, epsilon_decay, min_epsilon = 1.0, 0.99, 0.01
    rewards_history = []

    for _ in range(episodes):
        state = env.reset()
        action = select_action(state, q_table, epsilon)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = select_action(next_state, q_table, epsilon)

            # SARSA (On-policy): uses actual next action chosen
            td_target = reward + gamma * q_table[next_state[0], next_state[1], next_action]
            q_table[state[0], state[1], action] += alpha * (td_target - q_table[state[0], state[1], action])

            state = next_state
            action = next_action
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
    return q_table, rewards_history

def run_advanced_training():
    os.makedirs("outputs", exist_ok=True)
    env = CliffWalkingEnvironment()

    print("[INFO] Training Q-Learning Agent...")
    q_table_q, rewards_q = train_q_learning(env)

    print("[INFO] Training SARSA Agent...")
    q_table_sarsa, rewards_sarsa = train_sarsa(env)

    print("[SUCCESS] Both Algorithms Trained Successfully!")

    # 1. Save Comparison Graph
    plt.figure(figsize=(9, 5))
    plt.plot(rewards_q, label='Q-Learning (Off-Policy)', color='blue', alpha=0.8)
    plt.plot(rewards_sarsa, label='SARSA (On-Policy)', color='orange', alpha=0.8)
    plt.title("Algorithm Comparison: Q-Learning vs SARSA", fontsize=13, fontweight='bold')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/algorithm_comparison.png", dpi=300)
    plt.close()

    # 2. Generate and Save Path Frequency Heatmap
    state = env.reset()
    done = False
    visit_counts = np.zeros((env.height, env.width))
    visit_counts[state[0], state[1]] += 1

    while not done:
        action = np.argmax(q_table_q[state[0], state[1]])
        next_state, reward, done = env.step(action)
        state = next_state
        visit_counts[state[0], state[1]] += 1

    plt.figure(figsize=(10, 4))
    sns.heatmap(visit_counts, annot=True, cmap="YlGnBu", fmt="g", linewidths=0.5)
    plt.title("AI Agent Path Visit Heatmap (Q-Learning Policy)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/path_heatmap.png", dpi=300)
    plt.close()

    print("[SUCCESS] Advanced analytical charts (Comparison & Heatmap) saved in 'outputs/'!")
    return q_table_q

if __name__ == "__main__":
    run_advanced_training()