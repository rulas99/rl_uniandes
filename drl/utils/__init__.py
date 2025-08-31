import numpy as np
import matplotlib.pyplot as plt
import torch
from gridworld_env.GridWorld import GridWorldEnv

# Display animation of the agent in the environment
def display_animation(env, policy_net, num_episodes=1, size=8, device='cuda'):
    """
    Display animation of the trained agent in the environment.
    
    Args:
        env: The gymnasium environment
        policy_net: The trained DQN policy network
        num_episodes: Number of episodes to display
    """
    policy_net.eval()  # Set to evaluation mode
        
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        state_obs, info = env.reset(options={"agent_start": (size//2, size//2), "goal": (8,0)})
        state = GridWorldEnv.transform_obs(state_obs)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        step_count = 0
        done = False
        
        while not done:
            # Use the trained policy network to select action (greedy policy)
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
            
            # Take action in environment
            observation, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # Transform observation for next step
            if not done:
                observation = GridWorldEnv.transform_obs(observation)
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            step_count += 1
            
            # Render the environment
            env.render()
            
            # Safety check to avoid infinite loops
            if step_count > 300:
                print("Episode terminated after 300 steps")
                break
        
        print(f"Episode completed in {step_count} steps")
    
    env.close()


def plot_dqn_weights(model, save_path=None):
    """
    Función para visualizar las conexiones de la red DQN y sus pesos más relevantes.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get the weights for each layer
    layer1_weights = model.layer1.weight.data.cpu().numpy()  # Shape: (16, 2)
    layer2_weights = model.layer2.weight.data.cpu().numpy()  # Shape: (16, 16)
    layer3_weights = model.layer3.weight.data.cpu().numpy()  # Shape: (4, 16)
    
    # Layer 1: Input to Hidden (2 -> 16)
    im1 = axes[0].imshow(layer1_weights, cmap='RdBu', aspect='auto')
    axes[0].set_title('Layer 1: Input → Hidden\n(2 inputs → 16 neurons)')
    axes[0].set_xlabel('Input Features')
    axes[0].set_ylabel('Hidden Neurons')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Feature 1', 'Feature 2'])
    plt.colorbar(im1, ax=axes[0], label='Weight Value')
    
    # Add weight values as text
    for i in range(layer1_weights.shape[0]):
        for j in range(layer1_weights.shape[1]):
            axes[0].text(j, i, f'{layer1_weights[i,j]:.2f}', 
                        ha='center', va='center', fontsize=8)
    
    # Layer 2: Hidden to Hidden (16 -> 16)
    im2 = axes[1].imshow(layer2_weights, cmap='RdBu', aspect='auto')
    axes[1].set_title('Layer 2: Hidden → Hidden\n(16 → 16 neurons)')
    axes[1].set_xlabel('Input Hidden Neurons')
    axes[1].set_ylabel('Output Hidden Neurons')
    plt.colorbar(im2, ax=axes[1], label='Weight Value')
    
    # Layer 3: Hidden to Output (16 -> 4)
    im3 = axes[2].imshow(layer3_weights, cmap='RdBu', aspect='auto')
    axes[2].set_title('Layer 3: Hidden → Output\n(16 neurons → 4 actions)')
    axes[2].set_xlabel('Hidden Neurons')
    axes[2].set_ylabel('Output Actions')
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(['Action 0', 'Action 1', 'Action 2', 'Action 3'])
    plt.colorbar(im3, ax=axes[2], label='Weight Value')
    
    # Add weight values as text for output layer
    for i in range(layer3_weights.shape[0]):
        for j in range(layer3_weights.shape[1]):
            axes[2].text(j, i, f'{layer3_weights[i,j]:.2f}', 
                        ha='center', va='center', fontsize=6)
    
    plt.tight_layout()
    plt.show()
    
    # Print the most important connections
    print("\nMost relevant connections:")
    print("="*50)
    
    # Layer 1: Find strongest connections from inputs
    print("Layer 1 (Input → Hidden):")
    for input_idx in range(layer1_weights.shape[1]):
        strongest_neuron = np.argmax(np.abs(layer1_weights[:, input_idx]))
        strongest_weight = layer1_weights[strongest_neuron, input_idx]
        print(f"  Input {input_idx} → Hidden {strongest_neuron}: {strongest_weight:.3f}")
    
    # Layer 3: Find strongest connections to outputs
    print("\nLayer 3 (Hidden → Output):")
    for action_idx in range(layer3_weights.shape[0]):
        strongest_neuron = np.argmax(np.abs(layer3_weights[action_idx, :]))
        strongest_weight = layer3_weights[action_idx, strongest_neuron]
        print(f"  Hidden {strongest_neuron} → Action {action_idx}: {strongest_weight:.3f}")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')