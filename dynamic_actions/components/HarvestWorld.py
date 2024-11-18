import numpy as np
from numpy.random import choice, random
import matplotlib.pyplot as plt
from .Agent import Agent
from typing import List

from .GridWorldCell import (WaterCell, GrassCell, TreeAppleCell, 
                            RespawnCell, PrincipalRespawnCell)

AGENT_COLORS = ['#b510a7','#f28424','#05fcaa',
                '#f51bd4','#eeff03','#033dff']

class HarvestWorld():
    def __init__(self, ascii_map:str, regrowth_probs:List[float], num_agents:int=0):
        # max number of agents equal to 5
        if num_agents > 6:
            raise ValueError('The maximum number of agents is 2.') 
        
        self.timestep = 0
        self.ASCII_MAP = ascii_map
        self.regrowth_probs = regrowth_probs
        self.ascii_matrix = np.array([list(row) for row in ascii_map.strip().split('\n')])
        self.object_map = self.create_object_map()
        self.appletree_positions = [(cell.x, cell.y) for cell in self.object_map.flatten()
                                    if isinstance(cell, TreeAppleCell)]
        self.respawn_positions = [(cell.x, cell.y) for cell in self.object_map.flatten()
                                    if isinstance(cell, RespawnCell)]
        # use respawn_positions to place agents
        self.agents = [Agent(x=xy[0], y=xy[1], 
                            world_knowledge=self.object_map,
                            color=AGENT_COLORS[n]) for n, xy in enumerate(self.respawn_positions[:num_agents])]   
        
        
    def create_object_map(self):
        object_map = np.empty(self.ascii_matrix.shape, dtype=object)
        for i in range(self.ascii_matrix.shape[0]):
            for j in range(self.ascii_matrix.shape[1]):
                cell_char = self.ascii_matrix[i, j]
                if cell_char == 'W':
                    object_map[i, j] = WaterCell(i, j)
                elif cell_char in {'A', 'T'}:
                    with_apple = cell_char == 'A'
                    object_map[i, j] = TreeAppleCell(i, j, with_apple)
                elif cell_char == 'P':
                    object_map[i, j] = RespawnCell(i, j)
                elif cell_char == 'Q':
                    object_map[i, j] = PrincipalRespawnCell(i, j)
                else:
                    object_map[i, j] = GrassCell(i, j)
                    
        return object_map
    
    
    def get_apple_positions(self):
        return [(x,y) for x,y in self.appletree_positions if self.object_map[x,y].with_apple]

    
    def plot_map(self, ax):
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_title(f'Timestep: {self.timestep} - Total Apples: {len(self.get_apple_positions())}')
        ax.imshow([[c.color for c in row] for row in self.object_map])

        apple_positions = self.get_apple_positions()
        
        if apple_positions:
            x_coords, y_coords = zip(*apple_positions)
            ax.scatter(y_coords, x_coords, color='red')
            
        if self.agents:
            x_coords, y_coords = zip(*[agent.current_state[:2] for agent in self.agents])
            ax.scatter(y_coords, x_coords, color=AGENT_COLORS[:len(self.agents)],
                    marker='^', s=200)
        
        
    def advance_timestep(self):
        self.timestep += 1
        
        apple_positions = self.get_apple_positions()
        for apple_tree in self.appletree_positions:
            # look for adyacent cells
            i, j = apple_tree
            adyacent_apples = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            adyacent_apples = sum([1 for x,y in adyacent_apples if (x,y) in apple_positions])
            if (apple_tree not in apple_positions) and adyacent_apples >= 1:
                regrowth = choice(self.regrowth_probs)
                if random() < regrowth:
                    self.object_map[apple_tree].with_apple = True
                    
        for agent in self.agents:
            agent.move()
        
        return self.timestep