class TactiFootError(Exception):
    """Base exception for public TactiFoot Vision failures."""


class ConfigurationError(TactiFootError):
    """Raised when runtime configuration is invalid or incomplete."""


class ModelArtifactNotFound(TactiFootError):
    """Raised when a required model checkpoint or config file is missing."""


class AdapterUnavailable(TactiFootError):
    """Raised when an optional third-party backend cannot be imported."""


class PipelineError(TactiFootError):
    """Raised when a pipeline run cannot complete."""
