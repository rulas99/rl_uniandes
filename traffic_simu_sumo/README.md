# Traffic Light Control with Reinforcement Learning

This project uses reinforcement learning (Q-learning) to optimize traffic light timing at an intersection, with the goal of reducing congestion.

## Prerequisites

- SUMO (Simulation of Urban MObility) - Version 1.18.0 or newer
- Python 3.6 or newer
- NumPy, Matplotlib
- TraCI (included with SUMO)

## Setup

1. Make sure SUMO is installed on your system.
2. Set the SUMO_HOME environment variable:
   ```bash
   export SUMO_HOME=/path/to/sumo
   ```
   - On Linux, SUMO is typically installed in `/usr/share/sumo` or `/opt/sumo`
   - On Windows, it's typically in `C:\Program Files (x86)\Eclipse\Sumo`

3. Install required Python packages:
   ```bash
   pip install numpy matplotlib
   ```

## SUMO Configuration Files

The simulation environment is defined by several XML files:

- `nodes.nod.xml`: Defines the nodes (intersections) in the network
- `edges.edg.xml`: Defines the roads (edges) connecting nodes
- `connections.con.xml`: Defines how lanes connect at intersections
- `tlLogic.add.xml`: Defines traffic light phases and timing
- `routes.rou.xml`: Defines vehicle routes and traffic flow
- `simulation.sumocfg`: Main configuration file that brings everything together

## The Q-Learning Agent

Our agent:
- **States**: Level of congestion on each incoming lane (C1 and C2)
- **Actions**: 
  - Prioritize C1 (longer green time for C1)
  - Prioritize C2 (longer green time for C2)
  - Balanced (equal green time for both)
- **Reward**: Negative sum of congestion levels (the less congestion, the better)

## Running the Agent

To train the Q-learning agent:
```bash
python traffic_rl_agent.py --train
```

To evaluate a trained agent:
```bash
python traffic_rl_agent.py --evaluate
```

## Key Parameters

The parameters used in the Q-learning algorithm can be adjusted in the script:

- `MAX_STATE`: Maximum congestion level (scale)
- `ALPHA`: Learning rate
- `GAMMA`: Discount factor
- `EPSILON`, `EPSILON_MIN`, `EPSILON_DECAY`: Exploration parameters
- `NUM_EPISODES`: Number of training episodes
- `STEPS_PER_EPISODE`: Maximum steps per episode

## Customizing the Environment

You can modify the SUMO configuration files to create different traffic scenarios:

- Change traffic flow rates in `routes.rou.xml`
- Modify road layout in `nodes.nod.xml` and `edges.edg.xml`
- Adjust traffic light phases in the `tlLogic` section of `simple.net.xml`

## Visualizing Results

After training, a plot of rewards over episodes will be saved as `training_progress.png`. The trained Q-table will be saved as `q_table.npy`.

## Extending the Project

Some ideas for extending this project:
- Add more congestion metrics (waiting time, queue length)
- Implement a more complex reward function
- Use deep Q-learning for larger state spaces
- Experiment with different traffic scenarios 