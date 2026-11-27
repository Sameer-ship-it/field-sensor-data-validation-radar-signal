"""Domain exceptions for predictable input and processing failures."""


class SensorDataError(Exception):
    """Base exception for this package."""


class DatasetReadError(SensorDataError):
    """Raised when a supported dataset cannot be decoded."""


class DatasetSchemaError(SensorDataError):
    """Raised when a dataset is missing required structure."""


class SignalProcessingError(SensorDataError):
    """Raised when signal parameters are invalid or processing fails."""
