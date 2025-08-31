import random
import numpy as np
from typing import Tuple

class PageHinkleyTest:
    def __init__(self, delta=0.01, threshold=300.0):
        """
        Parámetros:
          - delta: un pequeño valor (bias) que evita detectar cambios por fluctuaciones menores.
          - threshold: umbral que, al ser superado, indica que se ha detectado un cambio.
        """
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reinicia los acumulados y contadores del test."""
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.n = 0

    def update(self, x):
        """
        Actualiza el PH-Test con el valor x (en este caso, la recompensa acumulada del episodio)
        y retorna True si se detecta un cambio.
        """
        self.n += 1
        # Actualización incremental de la media
        self.mean = self.mean + (x - self.mean) / self.n
        # Se acumula la diferencia entre el valor actual, la media y el delta
        self.cumulative_sum += (self.mean - x - self.delta)
        # Se guarda el mínimo acumulado
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)
        # Si la diferencia acumulada (desde el mínimo) supera el umbral, se detecta un cambio
        
        if (self.cumulative_sum - self.min_cumulative_sum) > self.threshold:
            return True
        else:
            return False


class AdaptativeAgent():
    def __init__(
        self,
        initial_state: Tuple[int, int],
        actions: list,
        n_rows: int,
        n_cols: int,
        alpha: float = 0.1,    # Tasa de aprendizaje base
        gamma: float = 0.9,    # Factor de descuento
        min_epsilon: float = 0.1,
        decay_rate: float = 0.003,
        alpha_max: float = 0.5,  # Tasa de aprendizaje máxima en caso de gran error
        td_threshold: float = 0.1,  # Umbral para activar adaptación
    ):
        # Parámetros de Q-learning
        self.alpha = alpha              # Tasa base
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.alpha_max = alpha_max
        self.td_threshold = td_threshold
        self.effective_alpha = alpha
        
        # Configuración del entorno
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.actions = actions
        
        # Estado inicial
        self.initial_state = initial_state
        self.current_state = initial_state
        
        # Q-table (diccionario de diccionarios)
        self.q_knowledge = np.zeros((len(self.actions), self.n_rows, self.n_cols))
        
        # Contadores
        self.steps = 0
        
        # Aux
        self.map_ix_action = {i: a for i, a in enumerate(self.actions)}  
        self.map_action_ix = {a: i for i, a in enumerate(self.actions)}      
        
    
    def update_actions(self, actions: dict) -> None:
        """Actualiza las acciones disponibles"""
        # Reconfigura la Q-table agregando (append) nuevas acciones sobre la q_knowledge original
        if len(self.actions) < len(actions):
            new_actions = len(actions) - len(self.actions)
            self.q_knowledge = np.append(
                                        self.q_knowledge, 
                                        np.zeros((new_actions, 
                                                self.n_rows, self.n_cols)), 
                                        axis=0)
            
        self.actions = [i for i in range(len(actions))]
        self.map_ix_action = {i: a for i, a in enumerate(self.actions)}  
        self.map_action_ix = {a: i for i, a in enumerate(self.actions)}
    
    
    def get_best_action(self, state: Tuple[int, int]) -> Tuple[str, float]:
        """Selecciona la mejor acción usando explotación"""
        action = np.argmax(self.q_knowledge[:, state[0], state[1]])
        action = self.map_ix_action[action]
        q_value = np.max(self.q_knowledge[:, state[0], state[1]])
        return action, q_value
    
    def update_q_value(self, current_state, action, reward, next_state):
        ai = self.map_action_ix[action]
        s0, s1 = current_state
        s0_, s1_ = next_state

        current_q = self.q_knowledge[ai, s0, s1]
        best_next = np.max(self.q_knowledge[:, s0_, s1_])
        td_error  = reward + self.gamma * best_next - current_q

        # sigma((|td|-θ)) en [0,1]
        activation = 1.0 / (1.0 + np.exp(-(abs(td_error) - self.td_threshold)))
        self.effective_alpha = self.alpha + (self.alpha_max - self.alpha) * activation

        new_q = current_q + self.effective_alpha * td_error
        self.q_knowledge[ai, s0, s1] = new_q

        return new_q
    
    def choose_action(self, epsilon: float) -> str:
        """Selección de acción epsilon-greedy"""
        if random.random() < epsilon:
            return random.choice(self.actions)
        # Explotación
        best_action, _ = self.get_best_action(self.current_state)
        return best_action
    
    def epsilon_decay(self, episode: int) -> float:
        """Decaimiento exponencial de epsilon"""
        return self.min_epsilon + (1 - self.min_epsilon) * np.exp(-self.decay_rate * episode)
    
    
    def restart(self) -> None:
        """Reinicia el agente a su estado inicial sin borrar la Q-table (se conserva el conocimiento)"""
        self.current_state = self.initial_state
        self.steps = 0
        
    def __str__(self) -> str:
        return f"AdaptativeAgent at {self.current_state} with Q-values {self.q_knowledge[self.current_state]}"
    
    
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


import numpy as np
import random
import matplotlib.pyplot as plt
from traffic import TrafficEnv
from agent import PageHinkleyTest, AdaptativeAgent

# Parámetros del entorno
LAMBDA_ARRIVAL_C1 = 4       # Parámetro de la distribución de Poisson para llegadas
LAMBDA_ARRIVAL_C2 = 2       # Parámetro de la distribución de Poisson para llegadas
MAX_STATE = 10               # Número máximo de coches en el sistema

# Parámetros de Q-Learning
ALPHA = 0.1              # Tasa de aprendizaje
GAMMA = 0.9             # Factor de descuento
EPSILON = 1.0            # Tasa de exploración inicial
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
NUM_EPISODES = 10001
MAX_STEPS = 30


# Inicialización del entorno y parámetros
N_ROWS = MAX_STATE + 1
N_COLS = MAX_STATE + 1

EXPLOTATION = 0.2

# Inicialización del agente
agent = AdaptativeAgent(
    initial_state=(4, 4),
    actions=[0,1,2],
    n_rows=N_ROWS,
    n_cols=N_COLS,
    decay_rate=.001, # más grande => más errático
    alpha=ALPHA,
    alpha_max=0.15,
    td_threshold=20, # más pequeño => más sensible
    min_epsilon=EPSILON_MIN,
)


# Inicialización del Page-Hinkley Test
ph_test = PageHinkleyTest(threshold=1000)

# Variables para el seguimiento del entrenamiento
learning_curve = []
exploration_curve = []

change_detection = []

env = TrafficEnv(c1_lambda=LAMBDA_ARRIVAL_C1, 
                c2_lambda=LAMBDA_ARRIVAL_C2, 
                max_steps=MAX_STEPS, 
                max_state=MAX_STATE)
rewards_per_episode = []

episode_changes = [3000, 8000]


eps = 1
# Bucle de entrenamiento
for ep in range(NUM_EPISODES):
    state = tuple(env.reset())      # (c1, c2)
    agent.current_state = state
    total_reward = 0.0
    
    for step in range(MAX_STEPS):
        # 1) Elegir acción ε-greedy
        action = agent.choose_action(eps)

        # 2) Interactuar con el entorno
        next_state, reward, done, info = env.step(action)
        next_state = tuple(next_state)

        # 3) Guardar recompensa
        total_reward += reward

        # 4) Actualizar Q-value con tasa adaptativa
        agent.update_q_value(
            current_state=state,
            action=action,
            reward=reward,
            next_state=next_state
        )

        state = next_state
        agent.current_state = next_state

        if done:
            break
        
                
    # Decay de ε **por episodio**:
    eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
    
    if ph_test.update(total_reward):
        print(f"Cambio detectado en episodio {ep} con recompensa {total_reward}", eps)
        change_detection.append(ep)
        
        eps = 1 # Aumentar exploración
        # Definir nuevas acciones ---
        if len(env.actions) < 4:
            extra_actions = [
                {'C1_service': 7, 'C2_service':  3},  # muy prioritario a C1
                {'C1_service':  3, 'C2_service': 7},  # muy prioritario a C2
            ]

            # Concatenar al conjunto actual ---
            updated_actions = env.actions + extra_actions
            env.actions   = updated_actions
            agent.update_actions(updated_actions)
        
        ph_test.reset()
        
    rewards_per_episode.append(total_reward)
    
    if ep in episode_changes:
        if ep == episode_changes[0]:
            env.c1_lambda = 5
            env.c2_lambda = 7
        elif ep == episode_changes[1]:
            env.c1_lambda = 3
            env.c2_lambda = 1
        
            
    if ep % 500 == 0:
        avg_last = np.mean(rewards_per_episode[-500:])
        print(f"Episodio {ep:5d} — Reward medio últimos 500 = {avg_last:.3f} — ε={eps:.3f} - α={agent.effective_alpha:.3f}")
    
            
print("Reward medio global:", np.mean(rewards_per_episode))
print("Reward máximo:",       np.max(rewards_per_episode))
print("Reward mínimo:",       np.min(rewards_per_episode))


"""
Print Output:
Episodio     0 — Reward medio últimos 500 = -388.000 — ε=0.999 - α=0.148
Episodio   500 — Reward medio últimos 500 = -486.586 — ε=0.606 - α=0.101
Episodio  1000 — Reward medio últimos 500 = -405.192 — ε=0.367 - α=0.106
Episodio  1500 — Reward medio últimos 500 = -360.138 — ε=0.223 - α=0.100
Episodio  2000 — Reward medio últimos 500 = -345.312 — ε=0.135 - α=0.100
Episodio  2500 — Reward medio últimos 500 = -320.658 — ε=0.082 - α=0.102
Episodio  3000 — Reward medio últimos 500 = -313.468 — ε=0.050 - α=0.100
Cambio detectado en episodio 3003 con recompensa -728.0 0.04951384249760823
Episodio  3500 — Reward medio últimos 500 = -601.292 — ε=0.608 - α=0.100
Episodio  4000 — Reward medio últimos 500 = -542.174 — ε=0.369 - α=0.100
Episodio  4500 — Reward medio últimos 500 = -501.538 — ε=0.224 - α=0.150
Episodio  5000 — Reward medio últimos 500 = -476.454 — ε=0.136 - α=0.150
Episodio  5500 — Reward medio últimos 500 = -457.022 — ε=0.082 - α=0.100
Episodio  6000 — Reward medio últimos 500 = -443.142 — ε=0.050 - α=0.100
Episodio  6500 — Reward medio últimos 500 = -420.580 — ε=0.030 - α=0.100
Episodio  7000 — Reward medio últimos 500 = -416.886 — ε=0.018 - α=0.100
Episodio  7500 — Reward medio últimos 500 = -417.508 — ε=0.011 - α=0.100
Episodio  8000 — Reward medio últimos 500 = -409.448 — ε=0.010 - α=0.100
Episodio  8500 — Reward medio últimos 500 = -357.778 — ε=0.010 - α=0.100
Episodio  9000 — Reward medio últimos 500 = -356.646 — ε=0.010 - α=0.100
Episodio  9500 — Reward medio últimos 500 = -340.226 — ε=0.010 - α=0.100
Episodio 10000 — Reward medio últimos 500 = -334.150 — ε=0.010 - α=0.100
Reward medio global: -415.3071692830717
Reward máximo: -213.0
Reward mínimo: -770.0
"""

ma = moving_average(rewards_per_episode, 500)
episodios = np.arange(len(ma)) + 500

plt.figure(figsize=(12, 6))
plt.plot(episodios, ma, label='Continuous Adaptive Q-Learning', color='#d15bf5', linewidth=2, zorder=10)
plt.plot(episodios, ma_tradq, label='Traditional Q-Learning', color='#608bcc', linestyle='-', linewidth=2, zorder=1)

for v in episode_changes:
    plt.axvline(x=v, color='red', linestyle='--', linewidth=1, zorder=30, label='Concept Drift' if v == episode_changes[0] else "")
for i, c in enumerate(change_detection):
    plt.axvline(x=c, color='#78d683', linestyle='-', linewidth=3, zorder=10, label='Change Detected (PH-Test)' if i == 0 else "")

plt.xlabel('Episode', fontsize=14, labelpad=15)
plt.ylabel('Cumulative Mean Reward\n(last 500 episodes)', fontsize=14, labelpad=15)
plt.title('Learning Curve: Traffic Light Control Over Non-Stationary Congestion', fontsize=16, pad=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
# change zorder of legend so axvline red be below
plt.legend(fontsize=10, frameon=True, ncol=1, loc='lower left').set_zorder(50)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.savefig('traffic_learning_curve.png', dpi=300, bbox_inches='tight')

plt.show()