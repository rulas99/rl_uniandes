from numpy import array as np_array
from numpy import zeros as np_zeros
from numpy import nan as np_nan
from numpy.random import choice, random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class Agent:
    def __init__(self, x: int, y: int, world_knowledge:np_array,
                agent_life: int = 20, gamma: float = 0.9,
                actions: List[str] = ['up', 'down', 'left', 'right'],
                alpha: float = 0.1, epsilon: float = 0.4,
                color: Tuple[int, int, int] = (0, 0, 0)):
        
        self.current_state = (x, y)
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.color: Tuple[int, int, int] = color
        self.world_knowledge = world_knowledge
        self.n_rows, self.n_cols = world_knowledge.shape
        self.agent_life = agent_life
        self.q_knowledge: Dict[Tuple[int, int], Dict[str, float]] = {
            (i, j): {}
            for i in range(self.n_rows) for j in range(self.n_cols)
            }
    
    
    def get_next_state(self, base_state: Tuple[int, int], next_action: str) -> Tuple[int, int]:
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
    
    
    def get_best_action(self, state: Tuple[int, int])-> Tuple[str, float]:
        
        # Obtener el valor máximo conocido para las acciones posibles
        max_q_value = max(self.q_knowledge[state].get(a, 0) for a in self.actions)
        
        # Ordenar las acciones por el valor Q estimado
        best_actions = [a for a in self.actions if self.q_knowledge[state].get(a, 0) == max_q_value]
        best_action = choice(best_actions)
        
        return best_action, max_q_value
    
    
    def update_state_qvalue(self, next_state: Tuple[int, int], action: str, 
                            arbitrary_reward:float=None) -> float:
        if next_state == self.current_state:
            return self.q_knowledge[self.current_state].get(action, 0)
        
        # Valor Q actual para la acción
        q_a_st = self.q_knowledge[self.current_state].get(action, 0)
        
        # Valor Q máximo en el siguiente estado
        _, q_max_a_st1 = self.get_best_action(next_state)
        
        reward = arbitrary_reward if arbitrary_reward is not None else self.world_knowledge[next_state].reward
        
        # Actualización de Q-learning
        q = q_a_st + self.alpha * (reward + (self.gamma * q_max_a_st1) - q_a_st)
        
        self.q_knowledge[self.current_state][action] = q
        
        return q
    
    
    
    def choose_greedy_action(self) -> str:
        
        if random() < self.epsilon:
            possible_actions = [ 
                                a for a in self.actions
                                if self.get_next_state(self.current_state, a) != self.current_state
                                ]
            
            return choice(possible_actions)
        else:
            best_action, _ = self.get_best_action(self.current_state)
            
            return best_action
    
    
    
    def move(self)-> Tuple[int, int]:
        next_action = self.choose_greedy_action()
        if next_action is None:
            return self.current_state 
        
        new_state = self.get_next_state(self.current_state, next_action)
        
        if self.agent_life == 0:
            
            self.current_state = (1, 3)
            self.agent_life = 20
            
            return self.current_state
            
        if new_state != self.current_state:        
            
            self.update_state_qvalue(
                                    action=next_action, 
                                    next_state=new_state
                                    )
            
            self.current_state = new_state
            
            if self.world_knowledge[self.current_state].char == 'A':
                self.world_knowledge[self.current_state].with_apple = False
                self.agent_life += 1
                
            self.agent_life -= 1
        
        return self.current_state
    
    
    def plot_knowledge(self, ax)-> None:
        knowledge_map = np_zeros(self.world_knowledge.shape)
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_title(f'Life: {self.agent_life}')
        
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                actions = self.q_knowledge[(i,j)]
                if not actions:
                    knowledge_map[i, j] = np_nan
                    continue
                
                max_value = max(actions.values())
                knowledge_map[i, j] = max(actions.values())
                
                
                ax.text(j, i, f'{round(max_value,2)}', ha='center',
                        va='center', color='white') 
        
        ax.imshow(knowledge_map, cmap='viridis')