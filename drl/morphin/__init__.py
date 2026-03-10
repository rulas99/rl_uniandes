"""MORPHIN deep-RL experiments package.

Importing this package also registers the local GridWorld env.
"""

from .gridworld_env.GridWorld import GridWorldEnv

__all__ = ["GridWorldEnv"]
