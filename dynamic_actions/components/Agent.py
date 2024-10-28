from numpy import array as np_array
from numpy import zeros as np_zeros
from numpy import nan as np_nan
from numpy.random import choice, random, shuffle
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, x: int, y: int, world_knowledge:np_array,
                 actions: tuple[str, str, str, str] = ('up','down', 'left', 'right'),
                 alhpa: float = 0.1, gamma: float = 0.9, epsilon: float = 0.8,
                 color: tuple[int, int, int] = (0, 0, 0)):
        
        self.current_state = (x, y)
        self.actions = actions
        self.alpha = alhpa
        self.gamma = gamma
        self.epsilon = epsilon
        self.color: tuple[int, int, int] = color
        self.world_knowledge = world_knowledge
        self.n_rows, self.n_cols = world_knowledge.shape
        self.knowledge = {(i,j): {} for i in range(self.n_rows) for j in range(self.n_cols)}
        
    def get_next_state(self, base_state, next_action: str) -> tuple:
        if next_action == 'up':
            new_state = (max(base_state[0] - 1, 0), base_state[1])
        elif next_action == 'down':
            new_state = (min(base_state[0] + 1, self.n_rows - 1), base_state[1])
        elif next_action == 'left':
            new_state = (base_state[0], max(base_state[1] - 1, 0))
        elif next_action == 'right':
            new_state = (base_state[0], min(base_state[1] + 1, self.n_cols - 1))     
        else:
            return base_state
                  
        return new_state
        
        
    def get_best_action(self, state):
        possible_actions = [a for a in self.actions if self.get_next_state(state, a) != state]
        shuffle(possible_actions)
        
        if not possible_actions:
            return None, 0
        
        # Obtener el valor mínimo conocido para las acciones posibles
        min_value = min([self.knowledge[state].get(a, 0) for a in possible_actions], default=0)
        
        # Ordenar las acciones por el valor Q estimado
        actions = sorted([(a, self.knowledge[state].get(a, min_value + 0.1 * min_value)) 
                        for a in possible_actions], 
                        key=lambda x: x[1], reverse=True)
        
        return actions[0]

    def calculate_next_action_reward(self, next_state, action):
        if next_state == self.current_state:
            return self.knowledge[self.current_state].get(action, 0)
        
        # Valor Q actual para la acción
        q_a_st = self.knowledge[self.current_state].get(action, 0)
        
        # Valor Q máximo en el siguiente estado
        _, q_max_a_st1 = self.get_best_action(next_state)
        
        # Actualización de Q-learning
        q = q_a_st + self.alpha * (self.world_knowledge[next_state].reward + (self.gamma * q_max_a_st1) - q_a_st)
        
        return q
    
    def update_q_knowledge(self, action, next_state):
        print(f'Updating Q knowledge for {self.current_state} - {action} - {next_state}')
        print(self.calculate_next_action_reward(next_state, action))
  
        self.knowledge[self.current_state][action] = self.calculate_next_action_reward(next_state, action)
    
    def choose_greedy_action(self, epsilon):
        if random() < epsilon:
            return choice(self.actions)
        else:
            best_action, max_reward = self.get_best_action(self.current_state)
            #self.knowledge[self.current_state]['short_memory'].append(max_reward)
            #self.current_reward = max_reward
            
            return best_action
        
        
    def plot_knowledge(self, ax):
        knowledge_map = np_zeros(self.world_knowledge.shape)
        if ax is None:
            ax = plt.gca()
        ax.clear()
        #ax.set_title(f'Timestep: {self.timestep} - Total Apples: {len(self.get_apple_positions())}')

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                actions = self.knowledge[(i,j)]
                if not actions:
                    knowledge_map[i, j] = np_nan
                    continue
                
                max_value = max(actions.values())
                knowledge_map[i, j] = max(actions.values())
                
                ax.text(j, i, f'{round(max_value,2)}', ha='center',
                        va='center', color='white') 
                                
        ax.imshow(knowledge_map, cmap='viridis')
        
        
    def move(self):
        next_action = self.choose_greedy_action(self.epsilon)
        new_state = self.get_next_state(self.current_state, next_action)
        
        #print(f'Agent at {self.current_state} moving {next_action} to {new_x, new_y}')
        
        if new_state != self.current_state:        
            if self.world_knowledge[new_state].char == 'A':
                self.world_knowledge[new_state].with_apple = False
                
            self.update_q_knowledge(action=next_action, next_state=new_state)
                
            self.current_state = new_state
        
        return self.current_state