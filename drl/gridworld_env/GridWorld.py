from __future__ import annotations
from typing import Optional, Dict, Any, Iterable, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """
    GridWorld sencillo: el agente (A) se mueve en una grilla cuadrada hasta llegar al objetivo (T).

    Observación:
        Dict(
            agent:  np.ndarray(shape=(2,), dtype=int64) -> [x, y]
            target: np.ndarray(shape=(2,), dtype=int64) -> [x, y]
        )

    Acciones (Discrete(4)):
        0: derecha (+x), 1: arriba (+y), 2: izquierda (-x), 3: abajo (-y)

    Terminación: cuando agent == target.
    Truncation: controlado externamente con TimeLimit o max_episode_steps en el registro.

    Parámetros:
        size           : tamaño de la grilla (size x size)
        reward_scale   : recompensa al alcanzar el objetivo (default: 1.0)
        step_penalty   : penalización por paso si no se alcanzó el objetivo (default: 0.0)
        render_mode    : "human" o "ansi" (opcional)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        size: int = 5,
        reward_scale: float = 1.0,
        step_penalty: float = 0.0,
        render_mode: Optional[str] = None,
        obstacles: Optional[Iterable[Tuple[int, int]]] = None,
        holes: Optional[Iterable[Tuple[int, int]]] = None,
        invalid_move_penalty: float = 0.0,   # penalización al intentar entrar a un obstáculo
        hole_penalty: float = 0.0,           # castigo al caer en hueco (recompensa = -hole_penalty)
        block_on_obstacle: bool = True,      # si True, el agente no se mueve al chocar con obstáculo
    ):
        assert size >= 2, "size debe ser >= 2"
        self.size = int(size)
        self.reward_scale = float(reward_scale)
        self.step_penalty = float(step_penalty)
        self.render_mode = render_mode

        # Estado interno
        self._agent_location = np.array([-1, -1], dtype=np.int64)
        self._target_location = np.array([-1, -1], dtype=np.int64)
        
        self.episode_step = 0

        # Espacios (usar dtypes concretos de numpy)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int64),
                "target": spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int64),
            }
        )
        self.action_space = spaces.Discrete(4)

        # Mapeo acción->dirección (coordenadas cartesianas discretas)
        self._action_to_direction = {
            0: np.array([+1, 0], dtype=np.int64),  # derecha
            1: np.array([0, +1], dtype=np.int64),  # arriba
            2: np.array([-1, 0], dtype=np.int64),  # izquierda
            3: np.array([0, -1], dtype=np.int64),  # abajo
        }
        
        self.invalid_move_penalty = float(invalid_move_penalty)  # NEW
        self.hole_penalty = float(hole_penalty)                  # NEW
        self.block_on_obstacle = bool(block_on_obstacle)         # NEW

        # Mapas booleanos de obstáculos y huecos, indexados como [x, y]
        self._obstacles = np.zeros((self.size, self.size), dtype=bool)
        self._holes = np.zeros((self.size, self.size), dtype=bool)
        
        if obstacles is not None:
            for (x, y) in obstacles:
                assert 0 <= x < self.size and 0 <= y < self.size, "obstacle fuera de rango"
                self._obstacles[x, y] = True
        if holes is not None:
            for (x, y) in holes:
                assert 0 <= x < self.size and 0 <= y < self.size, "hole fuera de rango"
                self._holes[x, y] = True
        
        # Guardar copia de obstáculos/huecos originales para restaurar en cada reset
        self._original_obstacles = self._obstacles.copy()
        self._original_holes = self._holes.copy()
        
        # Validación opcional del render_mode
        if self.render_mode is not None:
            assert self.render_mode in self.metadata["render_modes"], (
                f"render_mode debe ser uno de {self.metadata['render_modes']}"
            )

    # -------- helpers --------
    def _is_obstacle(self, pos: np.ndarray) -> bool:  # NEW
        x, y = int(pos[0]), int(pos[1])
        return bool(self._obstacles[x, y])

    def _is_hole(self, pos: np.ndarray) -> bool:      # NEW
        x, y = int(pos[0]), int(pos[1])
        return bool(self._holes[x, y])

    def _sanitize_maps(self):  # Evita que start/goal estén bloqueados
        # Restaurar obstáculos originales primero
        self._obstacles[:] = self._original_obstacles
        self._holes[:] = self._original_holes
        # Luego limpiar solo las posiciones de agente y goal
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        tx, ty = int(self._target_location[0]), int(self._target_location[1])
        self._obstacles[ax, ay] = False
        self._holes[ax, ay] = False
        self._obstacles[tx, ty] = False
        self._holes[tx, ty] = False
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {"agent": self._agent_location.copy(), 
                "target": self._target_location.copy()}

    def _get_info(self) -> Dict[str, Any]:
        # Distancia Manhattan (útil para debugging/análisis; no usarla como entrada al agente salvo que así lo definas)
        dist = np.linalg.norm(self._agent_location - self._target_location, ord=1)
        return {"distance": float(dist)}

    # -------- API Gymnasium --------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)  # seeding correcto

        start = goal = None
        opt_obstacles = opt_holes = None
        if options is not None:
            start = options.get("agent_start", None)
            goal  = options.get("goal", None)
            opt_obstacles = options.get("obstacles", None)
            opt_holes = options.get("holes", None)
            
        if opt_obstacles is not None:
            self._obstacles[:] = False
            for (x, y) in opt_obstacles:
                assert 0 <= x < self.size and 0 <= y < self.size, "obstacle fuera de rango"
                self._obstacles[x, y] = True
            self._original_obstacles = self._obstacles.copy()  # Actualizar originales
        if opt_holes is not None:
            self._holes[:] = False
            for (x, y) in opt_holes:
                assert 0 <= x < self.size and 0 <= y < self.size, "hole fuera de rango"
                self._holes[x, y] = True
            self._original_holes = self._holes.copy()  # Actualizar originales

        # Agente
        if start is not None:
            start = np.array(start, dtype=np.int64)
            assert start.shape == (2,), "agent_start debe ser (x,y)"
            assert (0 <= start).all() and (start < self.size).all(), "agent_start fuera de rango"
            self._agent_location = start
        else:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)

        # Meta
        if goal is not None:
            goal = np.array(goal, dtype=np.int64)
            assert goal.shape == (2,), "goal debe ser (x,y)"
            assert (0 <= goal).all() and (goal < self.size).all(), "goal fuera de rango"
            self._target_location = goal
        else:
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(0, self.size, size=2, dtype=np.int64)

        self._sanitize_maps()
        
        observation = self._get_obs()
        #observation = self._get_transformed_obs()
        info = self._get_info()
        self.episode_step = 0
        
        return observation, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Acción inválida: {action}"

        direction = self._action_to_direction[int(action)]
        candidate = np.clip(self._agent_location + direction, 0, self.size - 1)

        hit_obstacle = False
        fell_in_hole = False

        if self._is_obstacle(candidate) or bool(np.array_equal(candidate, self._agent_location)):
            hit_obstacle = True
            if self.block_on_obstacle:
                # no te mueves, sólo penalización opcional
                new_pos = self._agent_location
            else:
                # permite pisar obstáculo (no recomendado)
                new_pos = candidate
        else:
            new_pos = candidate

        self._agent_location = new_pos

        # Chequeo de hueco
        if self._is_hole(self._agent_location):
            fell_in_hole = True

        # Terminación
        reached_goal = bool(np.array_equal(self._agent_location, self._target_location))
        terminated = reached_goal or fell_in_hole

        # Recompensa
        if reached_goal:
            reward = self.reward_scale
        elif fell_in_hole:
            reward = -self.hole_penalty
        else:
            # penalización por paso + penalización por intento inválido => penalizacion por quedarse en el mismo lugar
            reward = -self.step_penalty - (self.invalid_move_penalty if hit_obstacle else 0.0)

        observation = self._get_obs()
        info = self._get_info()
        # NEW ---- añade flags útiles a info
        info.update({"hit_obstacle": hit_obstacle, "fell_in_hole": fell_in_hole})

        self.episode_step += 1

        truncated = False
        return observation, float(reward), terminated, truncated, info

    # -------- rendering --------
    def render(self):
        if self.render_mode == "ansi":
            return self._ascii_board()
        elif self.render_mode == "human":
            self._render_frame()
        else:
            return None

    def _ascii_board(self) -> str:
        rows = []
        for y in range(self.size - 1, -1, -1):
            row = []
            for x in range(self.size):
                pos = np.array([x, y], dtype=np.int64)
                if np.array_equal(pos, self._agent_location):
                    if np.array_equal(pos, self._target_location):
                        row.append("@")
                    else:
                        row.append("A")
                elif np.array_equal(pos, self._target_location):
                    row.append("T")
                # dibuja obstáculo/hueco si la celda está libre de A/T
                elif self._obstacles[x, y]:
                    row.append("#")
                elif self._holes[x, y]:
                    row.append("O")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        result = "\n".join(rows) + "\n"
        if self.size <= 10:
            x_labels = " ".join(str(i) for i in range(self.size))
            result += f" {x_labels}\n"
        return result

    def _render_frame(self):
        print(self._ascii_board(), end="")

    def close(self):
        pass
    
    @staticmethod
    def transform_obs(obs, size: int) -> np.ndarray:
        agent_pos = obs["agent"] / (size - 1)
        target_pos = obs["target"] / (size - 1)
        return np.concatenate([agent_pos, target_pos]).astype(np.float32)
        
        
# ---------- Registro para gym.make / gym.make_vec ----------
gym.register(
    id="entropia/GridWorld-v0",
    entry_point=GridWorldEnv,
    nondeterministic=True,
    max_episode_steps=150,  # evita episodios infinitos por diseño
)