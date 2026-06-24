"""Runtime config query and hot-reload routes."""

from __future__ import annotations

import copy
import logging
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.dependencies import get_app_state
from core.config_loader import apply_cli_overrides

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["Config"])


class ConfigUpdateRequest(BaseModel):
    overrides: dict[str, Any]


def _sanitize(config: dict) -> dict:
    """Sanitize config by masking sensitive fields."""
    safe = copy.deepcopy(config)
    for section in safe.values():
        if isinstance(section, dict):
            for k in list(section.keys()):
                val = section[k]
                if isinstance(val, str) and any(
                    secret in k.lower()
                    for secret in ["key", "secret", "password", "token", "api"]
                ):
                    section[k] = "***"
    return safe


@router.get("/")
async def get_config() -> dict:
    """Return the currently loaded config (sanitized)."""
    state = get_app_state()
    return _sanitize(state.config)


@router.put("/")
async def update_config(body: ConfigUpdateRequest) -> dict:
    """Hot-update configuration parameters."""
    state = get_app_state()
    state.config = apply_cli_overrides(state.config, body.overrides)
    logger.info("Config hot-updated with overrides: %s", body.overrides)
    return {"message": "Config updated", "overrides": body.overrides}