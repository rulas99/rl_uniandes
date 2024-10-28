class GridWorldCell:
    """Base class for all cells in the grid world."""
    def __init__(self, x: int, y: int, char: str = ' ', color: tuple[int, int, int] = (0, 0, 0), reward: int = 0):
        self.x: int = x
        self.y: int = y
        self._char: str = char
        self._color: tuple[int, int, int] = color
        self._reward: int = reward

    @property
    def char(self) -> str:
        return self._char

    @char.setter
    def char(self, value: str):
        self._char = value

    @property
    def color(self) -> tuple[int, int, int]:
        return self._color

    @color.setter
    def color(self, value: tuple[int, int, int]):
        self._color = value

    @property
    def reward(self) -> int:
        return self._reward

    @reward.setter
    def reward(self, value: int):
        self._reward = value

    def __str__(self) -> str:
        description_str = (
            f'Type: {self.__class__.__name__} ({self.char} - {self.color})\n'
            f'Position: ({self.x}, {self.y})\n'
            f'Reward: {self.reward}\n'
        )
        return description_str


class TreeAppleCell(GridWorldCell):
    """Represents a tree that can have an apple."""

    def __init__(self, x: int, y: int, with_apple: bool = True):
        super().__init__(x, y)
        self.with_apple: bool = with_apple

    @property
    def char(self) -> str:
        return 'A' if self.with_apple else 'T'

    @property
    def color(self) -> tuple[int, int, int]:
        return (45, 135, 87) if self.with_apple else (40, 117, 76)

    @property
    def reward(self) -> int:
        return 1 if self.with_apple else 0

    def __str__(self) -> str:
        description_str = super().__str__()
        description_str += f'Apple: {self.with_apple}\n'
        return description_str


class WallCell(GridWorldCell):
    """Represents a wall cell that cannot be traversed."""
    def __init__(self, x: int, y: int):
        super().__init__(x=x, y=y, char='W', color=(157, 161, 159))


class GrassCell(GridWorldCell):
    """Represents a grass cell that can be traversed."""
    def __init__(self, x: int, y: int):
        super().__init__(x=x, y=y, char=' ', color=(201, 242, 203))


class RespawnCell(GridWorldCell):
    """Represents a respawn point for agents."""
    def __init__(self, x: int, y: int):
        super().__init__(x=x, y=y, char='P', color=(201, 242, 203))


class PrincipalRespawnCell(RespawnCell):
    """Represents the principal respawn point."""
    def __init__(self, x: int, y: int):
        super().__init__(x=x, y=y)
        self.char = 'Q'
        self.color = (223, 235, 54)
