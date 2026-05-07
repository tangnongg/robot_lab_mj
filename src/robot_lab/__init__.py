from pathlib import Path

ROBOT_LAB_SRC_PATH: Path = Path(__file__).parent

# Explicitly import tasks submodule to trigger task registration
from . import tasks
