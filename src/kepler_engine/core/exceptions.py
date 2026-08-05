"""Domain exceptions for the Kepler engine."""


class KeplerEngineError(Exception):
    """Base error for the Kepler engine."""


class ModelNotFoundError(KeplerEngineError):
    """Raised when a registered / champion model cannot be resolved."""


class DataIngestionError(KeplerEngineError):
    """Raised when a data source cannot be read or fails schema validation."""


class LeakageViolationError(KeplerEngineError):
    """Raised when training features include known target-leakage columns."""


class ExperimentNotFoundError(KeplerEngineError):
    """Raised when a job / experiment run_id is unknown."""
