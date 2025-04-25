import numpy as np
import random
import matplotlib.pyplot as plt
import gym
from gym import spaces



# Función de recompensa dinámica según el nivel de congestión
def compute_reward(c1, c2):
    # Si C1 está congestionado (mayor a 7) y es el que tiene mayor o igual congestión
    if c1 > 7 and c1 >= c2:
        return -(2 * c1 + c2)
    # Si C2 está congestionado (mayor a 7) y es el que tiene mayor congestión
    elif c2 > 7 and c2 > c1:
        return -(c1 + 2 * c2)
    # Si ambos carriles tienes igual congestión alta/baja
    else:
        return -(c1 + c2)


#def compute_reward(c1, c2):
    # castiga la congestión y premia el servicio
#    return - (c1 + c2)

class TrafficEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                max_steps,
                c1_lambda,
                c2_lambda,
                max_state,
                ):
        super().__init__()

        # parámetros
        self.c1_lambda = c1_lambda
        self.c2_lambda = c2_lambda
        self.max_state = max_state
        self.max_steps = max_steps
        self.service_time_C1 = 1
        self.service_time_C2 = 1

        # acciones: Discrete(3), index en self.actions
        # cada tupla es la capacidad de servicio (vehículos/time step)
        self.actions = [
            {'C1_service': 5, 'C2_service': 2},
            {'C1_service': 2, 'C2_service': 5},
            {'C1_service': 3, 'C2_service': 3},
        ]

        # espacios
        # observación = (c1, c2), cada uno en [0..max_state]
        self.observation_space = spaces.Box(
            low=0, high=max_state, shape=(2,), dtype=np.int32
        )
        # acción = 0,1,2
        self.action_space = spaces.Discrete(len(self.actions))

        # estado interno
        self.seed()
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        # reinicia congestión y contador de pasos
        self.c1 = random.randint(1, 3)
        self.c2 = random.randint(1, 3)
        self.step_count = 0
        return np.array([self.c1, self.c2], dtype=np.int32)
    
    @property
    def actions(self):
        return self._actions

    @actions.setter
    def actions(self, new_actions):
        self._actions = new_actions
        # ¡Aquí actualizamos automáticamente el espacio de acciones!
        self.action_space = spaces.Discrete(len(self._actions))

    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
        self.step_count += 1
        srv = self.actions[action]

        # --- 1) Servicio completo a C1 ---
        served_c1 = min(self.c1, srv['C1_service'])
        self.c1 -= served_c1

        # --- 2) Llegadas a C2 durante el servicio de C1 ---
        arr2_during = np.random.poisson(self.c2_lambda)
        self.c2 = min(self.c2 + arr2_during, self.max_state)

        # --- 3) Servicio completo a C2 ---
        served_c2 = min(self.c2, srv['C2_service'])
        self.c2 -= served_c2

        # --- 4) Llegadas a C1 durante el servicio de C2 ---
        arr1_during = np.random.poisson(self.c1_lambda)
        self.c1 = min(self.c1 + arr1_during, self.max_state)
        
        # --- 5) Término de penalización por “sobre-servicio”
        #    si capacity > served, significa que sobráste «tiempo» de semáforo
        waste_c1 = max(srv['C1_service'] - served_c1, 0)
        waste_c2 = max(srv['C2_service'] - served_c2, 0)
        # ajusta beta para controlar cuánto penalizas
        #beta = 0.5
        penalty = 3*(waste_c1 + waste_c2)

        # Recompensa y fin de episodio
        reward = compute_reward(self.c1, self.c2) - penalty
        done = (self.step_count >= self.max_steps)

        obs = np.array([self.c1, self.c2], dtype=np.int32)
        info = {
            'served_c1': served_c1,
            'arr1_during': arr1_during,
            'served_c2': served_c2,
            'arr2_during': arr2_during,
            'penalty': penalty,
        }
        return obs, reward, done, info

    def render(self, title=None):
        # reutiliza tu plot() para visualizar estado actual
        size = self.max_state + 2
        H, W = size, size
        mid = size // 2

        # crea matriz base
        grid = np.zeros((H, W), int)
        # carriles verticales (C1) en columnas mid-1, mid
        for r in range(H):
            if r < mid-1 or r > mid:
                grid[r, mid-1] = 1
                grid[r, mid]   = 1
        # carriles horizontales (C2) en filas mid-1, mid
        for c in range(W):
            if c < mid-1 or c > mid:
                grid[mid-1, c] = 2
                grid[mid,   c] = 2
        # intersección
        grid[mid-1:mid+1, mid-1:mid+1] = 3

        plt.figure(figsize=(6,6))
        plt.imshow(grid, cmap='gray', interpolation='nearest')

        # scatter dinámico
        # C1
        coords_c1 = []
        # 5 arriba
        for d in range(1, mid):
            for cc in (mid-1, mid):
                coords_c1.append((mid-1 - d, cc))
        # 5 abajo
        for d in range(1, mid):
            for cc in (mid-1, mid):
                coords_c1.append((mid + d, cc))
        for y,x in coords_c1[:self.c1]:
            plt.scatter(x, y, color='red', s=100)
        # C2
        coords_c2 = []
        # 5 derecha
        for d in range(1, mid):
            for rr in (mid-1, mid):
                coords_c2.append((rr, mid + d))
        # 5 izquierda
        for d in range(1, mid):
            for rr in (mid-1, mid):
                coords_c2.append((rr, mid-1 - d))
        for y,x in coords_c2[:self.c2]:
            plt.scatter(x, y, color='blue', s=100)

        plt.xticks([]); plt.yticks([])
        plt.title(f"Paso {self.step_count} — C1={self.c1}  C2={self.c2}"+title if title else "")
        plt.show()
