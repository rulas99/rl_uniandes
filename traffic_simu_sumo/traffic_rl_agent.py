#!/usr/bin/env python3

import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# We need to import traci for SUMO interaction
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# Parameters for the Q-learning agent
MAX_STATE = 10          # Scale of 0 to 10 for C1 and C2 congestion
ALPHA = 0.1             # Learning rate
GAMMA = 0.95            # Discount factor
EPSILON = 1.0           # Initial exploration rate
EPSILON_MIN = 0.01      # Minimum exploration rate
EPSILON_DECAY = 0.995   # Decay rate for exploration
NUM_EPISODES = 100      # Number of episodes
STEPS_PER_EPISODE = 50  # Steps per episode

# Define the actions: Each action defines how long to keep each traffic light phase
ACTIONS = {
    0: {'name': 'Prioritize C1', 'phase_durations': [15, 4, 5, 4]},   # Longer green for C1
    1: {'name': 'Prioritize C2', 'phase_durations': [5, 4, 15, 4]},   # Longer green for C2
    2: {'name': 'Balanced', 'phase_durations': [10, 4, 10, 4]},       # Equal priority
}

class SUMOEnvironment:
    def __init__(self, sumo_cmd):
        self.sumo_cmd = sumo_cmd
        self.tl_id = "n2"  # Traffic light ID from your simulation
        self.lanes = {
            "C1": "e1_0",  # Lane for C1 (from west)
            "C2": "e3_0"   # Lane for C2 (from north)
        }
    
    def start(self):
        """Start the SUMO simulation"""
        traci.start(self.sumo_cmd)
    
    def reset(self):
        """Reset the simulation and return the initial state"""
        # Close previous simulation if it exists
        if traci.isLoaded():
            traci.close()
        
        # Start a new simulation
        traci.start(self.sumo_cmd)
        
        # Get initial state
        c1, c2 = self._get_congestion_levels()
        return (min(int(c1), MAX_STATE), min(int(c2), MAX_STATE))
    
    def step(self, action):
        """Apply action to the environment and return new state and reward"""
        # Apply the selected action by changing traffic light phase durations
        self._apply_action(action)
        
        # Run simulation for a few steps (each step is 1 second in SUMO)
        for _ in range(10):  # Simulate 10 seconds
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            traci.simulationStep()
        
        # Get new state
        c1, c2 = self._get_congestion_levels()
        new_state = (min(int(c1), MAX_STATE), min(int(c2), MAX_STATE))
        
        # Calculate reward (negative sum of congestion)
        reward = -(c1 + c2)
        
        # Check if simulation is done
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        return new_state, reward, done
    
    def close(self):
        """Close the simulation"""
        traci.close()
    
    def _get_congestion_levels(self):
        """Get congestion levels for both lanes (C1 and C2)"""
        try:
            # Get vehicle count as a simple measure of congestion
            # In a more advanced version, you could use lane occupancy or queue length
            c1 = traci.lane.getLastStepVehicleNumber(self.lanes["C1"])
            c2 = traci.lane.getLastStepVehicleNumber(self.lanes["C2"])
            
            # Optionally scale the values to match your MAX_STATE
            # For now, we'll return raw vehicle counts
            return c1, c2
        
        except traci.exceptions.TraCIException:
            # Return default values if there's an issue
            return 0, 0
    
    def _apply_action(self, action):
        """Apply the selected action by changing traffic light durations"""
        if action in ACTIONS:
            phase_durations = ACTIONS[action]['phase_durations']
            
            # Get the current program (should be '1' according to your XML)
            program = traci.trafficlight.getProgram(self.tl_id)
            
            # Set new phase durations
            for phase_index, duration in enumerate(phase_durations):
                traci.trafficlight.setPhase(self.tl_id, phase_index % 4)  # Set active phase
                traci.trafficlight.setPhaseDuration(self.tl_id, duration)  # Set duration
            
            # Could also completely set a new program with:
            # traci.trafficlight.setProgramLogic(self.tl_id, logic)
            # But that's more complex

def train_q_learning_agent():
    """Train a Q-learning agent in the SUMO environment"""
    # Initialize Q-table: state -> action -> value
    q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
    
    # Command to start SUMO with GUI (use "sumo" instead of "sumo-gui" for faster training without UI)
    sumo_cmd = ["sumo-gui", "-c", "simulation.sumocfg"]
    
    # Create environment
    env = SUMOEnvironment(sumo_cmd)
    
    # For plotting
    rewards_per_episode = []
    epsilon = EPSILON
    
    for episode in range(NUM_EPISODES):
        # Reset environment
        state = env.reset()
        total_reward = 0
        
        for step in range(STEPS_PER_EPISODE):
            # Choose action based on epsilon-greedy policy
            if random.random() < epsilon:
                action = random.randint(0, len(ACTIONS) - 1)  # Exploration
            else:
                action = np.argmax(q_table[state])  # Exploitation
            
            # Take action and observe new state and reward
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Update Q-value using Bellman equation
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += ALPHA * (
                reward + GAMMA * q_table[next_state][best_next_action] - q_table[state][action]
            )
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Record reward
        rewards_per_episode.append(total_reward)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}: Total reward = {total_reward:.2f}, Epsilon = {epsilon:.2f}")
    
    # Close the environment
    env.close()
    
    # Plot rewards over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Agent Performance in SUMO Traffic Environment")
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.show()
    
    # Save the Q-table
    np.save("q_table.npy", dict(q_table))
    
    return q_table

def evaluate_agent(q_table, num_episodes=5):
    """Evaluate a trained Q-learning agent"""
    sumo_cmd = ["sumo-gui", "-c", "simulation.sumocfg"]
    env = SUMOEnvironment(sumo_cmd)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        print(f"Starting evaluation episode {episode+1}...")
        time.sleep(1)  # Give time to see the GUI
        
        while not done and step < STEPS_PER_EPISODE:
            # Choose the best action based on Q-values
            action = np.argmax(q_table[state])
            
            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            print(f"Step {step}: State {state}, Action {ACTIONS[action]['name']}, Reward {reward:.2f}")
            
            state = next_state
            step += 1
            
            # Slow down for visualization
            time.sleep(0.5)
        
        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Traffic RL agent using Q-learning")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the agent")
    args = parser.parse_args()
    
    if args.train:
        print("Training Q-learning agent...")
        q_table = train_q_learning_agent()
        print("Training complete!")
    
    if args.evaluate:
        print("Evaluating trained agent...")
        # Load Q-table if it exists
        try:
            q_table = np.load("q_table.npy", allow_pickle=True).item()
            evaluate_agent(q_table)
        except FileNotFoundError:
            print("Error: No trained Q-table found. Run with --train first.")
    
    if not args.train and not args.evaluate:
        print("Please specify --train or --evaluate")
        print("Example: python traffic_rl_agent.py --train") 