from abc import ABC, abstractmethod
from typing import Literal

import wandb


class Logger(ABC):
    @abstractmethod
    def init(self, project_name: str, config: dict):
        pass

    @abstractmethod
    def log(self, data: dict, step: int = None):
        pass

    @abstractmethod
    def watch(self, model, criterion, log: str, log_freq: int):
        pass


class WandbLogger(Logger):
    def init(self, project_name: str, config: dict):
        wandb.init(project=project_name, config=config)

    def log(self, data: dict, step: int = None):
        wandb.log(data, step=step)

    def watch(self, model, criterion, log: Literal["gradients", "parameters", "all"] | None, log_freq: int):
        wandb.watch(model, criterion, log=log, log_freq=log_freq)


class ConsoleLogger(Logger):
    def init(self, project_name: str, config: dict):
        pass

    def log(self, data: dict, step: int = None):
        print(f"Step {step}: {data}")

    def watch(self, model, criterion, log: str, log_freq: int):
        pass


# (Create a context manager for the Logger)
class LoggerContext:
    def __init__(self, logger: Logger, project_name: str, config: dict):
        self.logger = logger
        self.project_name = project_name
        self.config = config

    def __enter__(self):
        self.logger.init(self.project_name, self.config)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Error occurred: {exc_type}, {exc_val}, {exc_tb}")
        if isinstance(self.logger, WandbLogger):
            wandb.finish()
