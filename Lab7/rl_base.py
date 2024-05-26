from abc import ABC, abstractmethod
from typing import List


class State(int):
    pass


class Action(int):
    pass


class Env(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def reset_env(self) -> State:
        pass

    @abstractmethod
    def render(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def step(self, action: Action) -> (State, float, bool, List[str], bool, bool):
        pass


class Agent(ABC):

    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def choose_action(self, state: State) -> Action:
        pass

    @abstractmethod
    def update_action_policy(self) -> None:
        pass

    @abstractmethod
    def learn(self, state: State, action: Action, reward: float, new_state: State, done: bool):
        pass

    @abstractmethod
    def save(self, save_path: str):
        pass

    @abstractmethod
    def get_instruction_string(self) -> List[str]:
        pass
