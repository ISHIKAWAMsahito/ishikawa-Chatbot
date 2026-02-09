import jwt
import logging
from datetime import datetime, timezone
# ★重要: config.py で一元管理された SECRET_KEY をインポート
from core.config import SECRET_KEY

logger = logging.getLogger(__name__)

# アルゴリズム固定
ALGORITHM = "HS256"

def validate_ws_token(token: str) -> bool:
    """
    WebSocket接続用の一時トークンを検証する。
    Args:
        token (str): クライアントから送信されたJWT
    Returns:
        bool: 有効な場合 True
    """
    if not token:
        logger.warning("WebSocket auth failed: Token is missing.")
        return False
    
    try:
        # core.config.SECRET_KEY を使用してデコード
        # PyJWT の decode は署名検証と期限切れチェックを自動で行う
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return True

    except jwt.ExpiredSignatureError:
        logger.warning("WebSocket auth failed: Token has expired.")
        return False
    except jwt.InvalidTokenError as e:
        # 署名不一致 (InvalidSignatureError) もここに含まれる
        logger.warning(f"WebSocket auth failed: Invalid token or Signature mismatch. Details: {e}")
        return False
    except Exception as e:
        logger.error(f"WebSocket auth system error: {e}", exc_info=True)
        return False