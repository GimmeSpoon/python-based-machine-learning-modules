import multiprocessing as mp

mp.set_start_method("spawn")

__name__
__version__ = "0.0.1"

__all__ = [
    'model',
    'trainer',
    'interface',
    'visualizer',
    'remote',
    'tester',
    'logger',
]