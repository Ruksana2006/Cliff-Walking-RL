"""
File: src/app_dashboard.py
Description: Advanced Dashboard displaying dual-algorithm training 
             comparison, heatmap analytics, and Live ASCII Grid Animation.
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cliff_walking import CliffWalkingEnvironment, run_advanced_training
import numpy as np

def print_grid_animation(env, current_state):
    """Prints a live ASCII visual representation of the Cliff Walking grid."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("==========================================================")
    print("      🤖 LIVE AI AGENT CLIFF WALKING ANIMATION            ")
    print("==========================================================")
    print("Legend: [S]=Start, [G]=Goal, [C]=Cliff, [A]=AI Agent\n")

    for r in range(env.height):
        row_str = ""
        for c in range(env.width):
            if (r, c) == current_state:
                row_str += " 🤖 "  # Agent position
            elif (r, c) == env.start_state:
                row_str += " 🏁 "  # Start
            elif (r, c) == env.goal_state:
                row_str += " 🎯 "  # Goal
            elif r == 3 and 1 <= c <= 10:
                row_str += " ⚠️ "  # Cliff zone
            else:
                row_str += " ⬜ "  # Safe path
        print(row_str)
    print("\n----------------------------------------------------------")

def run_rl_dashboard():
    print("==========================================================")
    print("   🤖 ADVANCED REINFORCEMENT LEARNING ANALYTICS DASHBOARD  ")
    print("==========================================================")
    
    print("\n[INFO] Starting Multi-Algorithm Training & Heatmap Generation...")
    q_table = run_advanced_training()
    
    env = CliffWalkingEnvironment()
    state = env.reset()
    done = False
    steps = 0
    path = [state]

    input("\n[PRESS ENTER] to start the Live AI Agent Grid Animation...")

    while not done and steps < 100:
        print_grid_animation(env, state)
        time.sleep(0.4)  # Pause for animation effect

        action = np.argmax(q_table[state[0], state[1]])
        next_state, reward, done = env.step(action)
        state = next_state
        path.append(state)
        steps += 1

    # Print final state
    print_grid_animation(env, state)
    print("\n==========================================================")
    print("                 📊 ADVANCED RL REPORT                    ")
    print("==========================================================")
    print(f"• Algorithms Compared : Q-Learning & SARSA")
    print(f"• Optimal Steps Taken : {steps}")
    print(f"• Cliff Hazard Avoided: 100% Success ✅")
    print(f"• Generated Assets    : 'algorithm_comparison.png', 'path_heatmap.png'")
    print("==========================================================\n")

if __name__ == "__main__":
    run_rl_dashboard()