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
    0: {'name': 'Strongly Prioritize C1', 'phase_durations': [20, 4, 5, 4]},   # Much longer green for C1
    1: {'name': 'Moderately Prioritize C1', 'phase_durations': [15, 4, 7, 4]}, # Longer green for C1
    2: {'name': 'Balanced', 'phase_durations': [10, 4, 10, 4]},                # Equal priority
    3: {'name': 'Moderately Prioritize C2', 'phase_durations': [7, 4, 15, 4]}, # Longer green for C2
    4: {'name': 'Strongly Prioritize C2', 'phase_durations': [5, 4, 20, 4]},   # Much longer green for C2
}

class SUMOEnvironment:
    def __init__(self, sumo_cmd):
        self.sumo_cmd = sumo_cmd
        self.tl_id = "n2"  # Traffic light ID from your simulation
        self.lanes = {
            "C1": "e1_0",  # Lane for C1 (from west)
            "C2": "e3_0"   # Lane for C2 (from north)
        }
        
        # Metrics history for tracking performance
        self.waiting_time_history = []
        self.queue_length_history = []
        self.vehicle_count_history = []
    
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
        
        # Reset metrics history
        self.waiting_time_history = []
        self.queue_length_history = []
        self.vehicle_count_history = []
        
        # Get initial state
        c1_state, c2_state = self._get_state()
        return (c1_state, c2_state)
    
    def step(self, action):
        """Apply action to the environment and return new state and reward"""
        # Store metrics before action for comparison
        metrics_before = self._get_all_metrics()
        
        # Apply the selected action by changing traffic light phase durations
        self._apply_action(action)
        
        # Run simulation for a few steps (each step is 1 second in SUMO)
        for _ in range(15):  # Simulate 15 seconds
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            traci.simulationStep()
        
        # Get new metrics after action
        metrics_after = self._get_all_metrics()
        
        # Get new state representation
        c1_state, c2_state = self._get_state()
        new_state = (c1_state, c2_state)
        
        # Calculate reward based on metrics change
        reward = self._calculate_reward(metrics_before, metrics_after)
        
        # Check if simulation is done
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        # Store metrics for history
        self.waiting_time_history.append((metrics_after['waiting_time_c1'], metrics_after['waiting_time_c2']))
        self.queue_length_history.append((metrics_after['queue_length_c1'], metrics_after['queue_length_c2']))
        self.vehicle_count_history.append((metrics_after['vehicle_count_c1'], metrics_after['vehicle_count_c2']))
        
        return new_state, reward, done
    
    def close(self):
        """Close the simulation"""
        if traci.isLoaded():
            traci.close()
    
    def _get_state(self):
        """Get a discrete state representation based on multiple metrics"""
        metrics = self._get_all_metrics()
        
        # Combine metrics to create a state representation
        # We'll use a weighted sum of normalized metrics to create a 0-10 state value
        c1_metrics = [
            metrics['vehicle_count_c1'] / 10,  # Normalize assuming max 10 vehicles
            metrics['queue_length_c1'] / 10,   # Normalize assuming max 10 queue length
            metrics['waiting_time_c1'] / 100   # Normalize assuming max 100s waiting time
        ]
        
        c2_metrics = [
            metrics['vehicle_count_c2'] / 10,
            metrics['queue_length_c2'] / 10,
            metrics['waiting_time_c2'] / 100
        ]
        
        # Calculate weighted sum with more emphasis on waiting time and queue length
        c1_state = min(int(0.3 * c1_metrics[0] + 0.3 * c1_metrics[1] + 0.4 * c1_metrics[2] * MAX_STATE), MAX_STATE)
        c2_state = min(int(0.3 * c2_metrics[0] + 0.3 * c2_metrics[1] + 0.4 * c2_metrics[2] * MAX_STATE), MAX_STATE)
        
        return c1_state, c2_state
    
    def _get_all_metrics(self):
        """Get various traffic metrics for both lanes"""
        try:
            # Get vehicle count
            vehicle_count_c1 = traci.lane.getLastStepVehicleNumber(self.lanes["C1"])
            vehicle_count_c2 = traci.lane.getLastStepVehicleNumber(self.lanes["C2"])
            
            # Get queue length (number of vehicles that are stopped)
            queue_length_c1 = traci.lane.getLastStepHaltingNumber(self.lanes["C1"])
            queue_length_c2 = traci.lane.getLastStepHaltingNumber(self.lanes["C2"])
            
            # Get total waiting time for all vehicles on each lane
            waiting_time_c1 = 0
            waiting_time_c2 = 0
            
            vehicles_c1 = traci.lane.getLastStepVehicleIDs(self.lanes["C1"])
            vehicles_c2 = traci.lane.getLastStepVehicleIDs(self.lanes["C2"])
            
            for vehicle_id in vehicles_c1:
                waiting_time_c1 += traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            
            for vehicle_id in vehicles_c2:
                waiting_time_c2 += traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            
            return {
                'vehicle_count_c1': vehicle_count_c1,
                'vehicle_count_c2': vehicle_count_c2,
                'queue_length_c1': queue_length_c1,
                'queue_length_c2': queue_length_c2,
                'waiting_time_c1': waiting_time_c1,
                'waiting_time_c2': waiting_time_c2
            }
        
        except traci.exceptions.TraCIException:
            # Return default values if there's an issue
            return {
                'vehicle_count_c1': 0,
                'vehicle_count_c2': 0,
                'queue_length_c1': 0,
                'queue_length_c2': 0,
                'waiting_time_c1': 0,
                'waiting_time_c2': 0
            }
    
    def _calculate_reward(self, metrics_before, metrics_after):
        """Calculate reward based on changes in traffic metrics"""
        # We want to reward reduction in waiting time and queue length
        
        # Changes in waiting time (negative is good)
        waiting_time_change_c1 = metrics_after['waiting_time_c1'] - metrics_before['waiting_time_c1']
        waiting_time_change_c2 = metrics_after['waiting_time_c2'] - metrics_before['waiting_time_c2']
        
        # Changes in queue length (negative is good)
        queue_change_c1 = metrics_after['queue_length_c1'] - metrics_before['queue_length_c1']
        queue_change_c2 = metrics_after['queue_length_c2'] - metrics_before['queue_length_c2']
        
        # Overall congestion level (negative is good)
        total_vehicles = metrics_after['vehicle_count_c1'] + metrics_after['vehicle_count_c2']
        total_waiting_time = metrics_after['waiting_time_c1'] + metrics_after['waiting_time_c2']
        total_queue = metrics_after['queue_length_c1'] + metrics_after['queue_length_c2']
        
        # Calculate advanced reward with dynamic priorities
        # Prioritize improvements in which lane is more congested
        c1_congestion = metrics_after['queue_length_c1'] + 0.1 * metrics_after['waiting_time_c1']
        c2_congestion = metrics_after['queue_length_c2'] + 0.1 * metrics_after['waiting_time_c2']
        
        # Weight changes based on congestion level
        total_congestion = max(1, c1_congestion + c2_congestion)  # Avoid division by zero
        c1_weight = c1_congestion / total_congestion
        c2_weight = c2_congestion / total_congestion
        
        # Weighted changes (more weight to more congested lane)
        weighted_waiting_change = -(c1_weight * waiting_time_change_c1 + c2_weight * waiting_time_change_c2)
        weighted_queue_change = -(c1_weight * queue_change_c1 + c2_weight * queue_change_c2)
        
        # Final reward combines multiple factors
        # We use negative values because lower congestion is better
        reward = (
            -0.5 * total_waiting_time         # Overall waiting time
            -2.0 * total_queue                # Overall queue length (weighted higher)
            +5.0 * weighted_waiting_change    # Improvement in waiting time
            +10.0 * weighted_queue_change     # Improvement in queue length
        )
        
        return reward
    
    def _apply_action(self, action):
        """Apply the selected action by changing traffic light durations"""
        if action in ACTIONS:
            phase_durations = ACTIONS[action]['phase_durations']
            
            # Get the current program
            program = traci.trafficlight.getProgram(self.tl_id)
            
            # Set new phase durations
            for phase_index, duration in enumerate(phase_durations):
                traci.trafficlight.setPhase(self.tl_id, phase_index % 4)  # Set active phase
                traci.trafficlight.setPhaseDuration(self.tl_id, duration)  # Set duration
    
    def plot_metrics_history(self, save_path=None):
        """Plot the history of traffic metrics"""
        if not self.waiting_time_history:
            print("No metrics history available to plot")
            return
        
        # Extract data for plotting
        time_steps = range(len(self.waiting_time_history))
        waiting_c1 = [wt[0] for wt in self.waiting_time_history]
        waiting_c2 = [wt[1] for wt in self.waiting_time_history]
        queue_c1 = [ql[0] for ql in self.queue_length_history]
        queue_c2 = [ql[1] for ql in self.queue_length_history]
        vehicles_c1 = [vc[0] for vc in self.vehicle_count_history]
        vehicles_c2 = [vc[1] for vc in self.vehicle_count_history]
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot waiting times
        axs[0].plot(time_steps, waiting_c1, 'r-', label='C1 Waiting Time')
        axs[0].plot(time_steps, waiting_c2, 'b-', label='C2 Waiting Time')
        axs[0].set_title('Total Waiting Time')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Waiting Time (s)')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot queue lengths
        axs[1].plot(time_steps, queue_c1, 'r-', label='C1 Queue Length')
        axs[1].plot(time_steps, queue_c2, 'b-', label='C2 Queue Length')
        axs[1].set_title('Queue Length')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Number of Stopped Vehicles')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot vehicle counts
        axs[2].plot(time_steps, vehicles_c1, 'r-', label='C1 Vehicles')
        axs[2].plot(time_steps, vehicles_c2, 'b-', label='C2 Vehicles')
        axs[2].set_title('Vehicle Count')
        axs[2].set_xlabel('Time Step')
        axs[2].set_ylabel('Number of Vehicles')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def train_q_learning_agent():
    """Train a Q-learning agent in the SUMO environment"""
    # Initialize Q-table: state -> action -> value
    q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))
    
    # Command to start SUMO (use "sumo" instead of "sumo-gui" for faster training without UI)
    sumo_cmd = ["sumo", "-c", "simulation.sumocfg"]  # No GUI for faster training
    
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
    plt.savefig("training_progress_advanced.png")
    plt.show()
    
    # Save the Q-table
    np.save("q_table_advanced.npy", dict(q_table))
    
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
        
        # Plot metrics for this episode
        env.plot_metrics_history(f"metrics_episode_{episode+1}.png")
    
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Traffic RL agent using Q-learning")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the agent")
    args = parser.parse_args()
    
    if args.train:
        print("Training advanced Q-learning agent...")
        q_table = train_q_learning_agent()
        print("Training complete!")
    
    if args.evaluate:
        print("Evaluating trained agent...")
        # Load Q-table if it exists
        try:
            q_table = np.load("q_table_advanced.npy", allow_pickle=True).item()
            evaluate_agent(q_table)
        except FileNotFoundError:
            print("Error: No trained Q-table found. Run with --train first.")
    
    if not args.train and not args.evaluate:
        print("Please specify --train or --evaluate")
        print("Example: python traffic_rl_advanced.py --train") 