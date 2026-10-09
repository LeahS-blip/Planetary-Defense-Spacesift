"""SpaceSift: injection-recovery experiments for transit detectability."""

__version__ = "0.1.0"

# Bump whenever detrending, binning, or injection logic changes in a way that
# could alter results. Recorded in every experiment's record.json.
PREPROCESSING_VERSION = "0.1"


def json_safe(obj):
    """NaN/inf -> None, recursively. Python's json writes NaN, which is not valid JSON
    and which browsers refuse to parse."""
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
