from tactifoot_vision.domain import AdapterUnavailable


class BoTSORTTracker:
    """Reserved BoTSORT facade.

    The current repository does not provide a stable standalone BoTSORT package API,
    so this backend is intentionally unavailable instead of silently falling back.
    """

    def __init__(self, *, frame_rate: int = 25) -> None:
        _ = frame_rate
        raise AdapterUnavailable(
            "BoTSORT tracking is disabled until a stable production adapter is selected."
        )
