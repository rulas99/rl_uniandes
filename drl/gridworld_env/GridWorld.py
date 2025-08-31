from __future__ import annotations
from typing import Optional, Dict, Any
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
    ):
        assert size >= 2, "size debe ser >= 2"
        self.size = int(size)
        self.reward_scale = float(reward_scale)
        self.step_penalty = float(step_penalty)
        self.render_mode = render_mode

        # Estado interno
        self._agent_location = np.array([-1, -1], dtype=np.int64)
        self._target_location = np.array([-1, -1], dtype=np.int64)

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

        # Validación opcional del render_mode
        if self.render_mode is not None:
            assert self.render_mode in self.metadata["render_modes"], (
                f"render_mode debe ser uno de {self.metadata['render_modes']}"
            )

    # -------- helpers --------
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

        start = None
        goal = None
        if options is not None:
            start = options.get("agent_start", None)
            goal  = options.get("goal", None)

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

        observation = self._get_obs()
        #observation = self._get_transformed_obs()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, info

    def step(self, action: int):
        # Validación de acción
        assert self.action_space.contains(action), f"Acción inválida: {action}"

        # Movimiento con límites (no sale de la grilla)
        direction = self._action_to_direction[int(action)]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Terminación si alcanza el objetivo
        terminated = bool(np.array_equal(self._agent_location, self._target_location))
        truncated = False  # usa TimeLimit o max_episode_steps en el registro si quieres cortar por tiempo

        # Recompensa: shaped mínima (éxito vs. penalización por paso)
        if terminated:
            reward = self.reward_scale
        else:
            reward = -self.step_penalty

        observation = self._get_obs()
        #observation = self._get_transformed_obs()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        # Gymnasium espera reward como float nativo
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
        """Generate ASCII representation of the grid world."""
        rows = []
        # Print from top to bottom (y descending) to match standard grid visualization
        for y in range(self.size - 1, -1, -1):
            row = []
            for x in range(self.size):
                pos = np.array([x, y], dtype=np.int64)
                if np.array_equal(pos, self._agent_location):
                    if np.array_equal(pos, self._target_location):
                        # Agent reached target - show both
                        row.append("@")  # or "X" to indicate completion
                    else:
                        row.append("A")
                elif np.array_equal(pos, self._target_location):
                    row.append("T")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        
        # Add coordinate labels for better debugging
        result = "\n".join(rows) + "\n"
        if self.size <= 10:  # Only show coordinates for small grids
            # Add x-axis labels
            x_labels = " ".join(str(i) for i in range(self.size))
            result += f" {x_labels}\n"
        
        return result

    def _render_frame(self):
        print(self._ascii_board(), end="")

    def close(self):
        pass
    
    @staticmethod
    def transform_obs(obs: Dict[str, np.ndarray], size : int = 8) -> np.ndarray:
        # Regresar solo la posicion del agente normalizada [0,1] y en int64
        #return (obs["agent"]-obs["target"]) / size
        return obs["agent"] / size
        
        
# ---------- Registro para gym.make / gym.make_vec ----------
gym.register(
    id="entropia/GridWorld-v0",
    entry_point=GridWorldEnv,
    nondeterministic=True,
    max_episode_steps=300,  # evita episodios infinitos por diseño
)