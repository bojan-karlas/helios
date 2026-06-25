"""Models package: shared base contract + registry."""

from helios.models.base import Model, get_model, register_model, registered_models

__all__ = ["Model", "get_model", "register_model", "registered_models"]
