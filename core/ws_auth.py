"""
WebSocket 接続用の短期トークン管理（管理者向け /ws/settings の認証用）。
トークンは 60 秒間有効。main.py の WebSocket は ?token=xxx で検証する。
"""
import secrets
import time
import logging

logger = logging.getLogger(__name__)

# token -> 有効期限 (Unix timestamp)
_ws_tokens: dict[str, float] = {}
_WS_TOKEN_TTL_SEC = 60


def create_ws_token() -> str:
    """管理者認証済みセッション用の短期トークンを発行する。"""
    token = secrets.token_urlsafe(32)
    _ws_tokens[token] = time.time() + _WS_TOKEN_TTL_SEC
    _prune_expired()
    return token


def validate_ws_token(token: str | None) -> bool:
    """トークンが存在し未期限切れなら True。使用後は削除しない（同一トークンで再接続を許す）。"""
    if not token or not token.strip():
        return False
    _prune_expired()
    expiry = _ws_tokens.get(token.strip())
    if expiry is None:
        return False
    if time.time() > expiry:
        del _ws_tokens[token.strip()]
        return False
    return True


def _prune_expired() -> None:
    """期限切れトークンを削除する。"""
    now = time.time()
    expired = [t for t, e in _ws_tokens.items() if e <= now]
    for t in expired:
        del _ws_tokens[t]
