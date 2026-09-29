from enum import Flag, auto


class PipelineMode(Flag):
    """
    PipelineMode determines what parts of the pipeline execute.
    - SCENE skips refinement entirely.
    - REFINER allows skipping rendering by passing a cached render.
    - JOINT runs the full pipeline.
    - EVAL runs the full pipeline for test-time evaluation.
    """
    SCENE = auto()
    REFINER = auto()
    JOINT = SCENE | REFINER
    EVAL = auto()
    ALL = SCENE | REFINER | EVAL
