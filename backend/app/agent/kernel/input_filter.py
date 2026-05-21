"""从 ``config_json`` 解析 ``InputContentFilterConfig``（与 :mod:`app.agent.kernel.spec` 中的模型配搭）。"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.kernel.spec import InputContentFilterConfig

logger = logging.getLogger(__name__)

_DEFAULT = InputContentFilterConfig()


def parse_input_content_filter_from_config(config_json: Any) -> InputContentFilterConfig:
    if not config_json or not isinstance(config_json, dict):
        return _DEFAULT
    raw = config_json.get("input_filter")
    if not isinstance(raw, dict):
        return _DEFAULT

    def _as_bool(x: object, default: bool) -> bool:
        if x is None:
            return default
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return x != 0
        s = str(x).strip().lower() if str(x) else ""
        if s in ("1", "true", "yes", "on", "y"):
            return True
        if s in ("0", "false", "no", "off", "n", ""):
            return False
        return default

    enabled = _as_bool(raw.get("enabled", False), False)
    kws: list[str] = []
    kr = raw.get("banned_keywords")
    if isinstance(kr, list):
        for x in kr:
            if isinstance(x, str) and (s := x.strip()):
                kws.append(s)
    pats: list[str] = []
    pr = raw.get("banned_regex")
    if isinstance(pr, list):
        for x in pr:
            if isinstance(x, str) and (s := x.strip()):
                pats.append(s)
    max_c = raw.get("max_user_chars")
    mci: int | None
    if max_c is None:
        mci = None
    else:
        try:
            mci = int(max_c)
        except (TypeError, ValueError):
            mci = None
            logger.warning("input_filter.max_user_chars 非整数，已忽略 | raw=%r", max_c)
        else:
            if mci < 0:
                mci = None
    tr = _as_bool(raw.get("truncate_on_max", False), False)
    rej = raw.get("reject_message")
    reject = (
        rej.strip() if isinstance(rej, str) and rej.strip() else _DEFAULT.reject_message
    ) or _DEFAULT.reject_message
    return InputContentFilterConfig(
        enabled=enabled,
        banned_keywords=tuple(kws),
        banned_regex=tuple(pats),
        max_user_chars=mci,
        truncate_on_max=tr,
        reject_message=reject,
    )
