import jwt
import logging
from datetime import datetime, timezone
from core.config import SECRET_KEY

logger = logging.getLogger(__name__)

# アルゴリズムを固定 (system.py と一致させる)
ALGORITHM = "HS256"

def validate_ws_token(token: str) -> bool:
    """
    WebSocket接続用の一時トークンを検証する。
    """
    if not token:
        logger.warning("WebSocket auth failed: Token is missing.")
        return False
    
    try:
        # core.config.SECRET_KEY を使用してデコード
        # system.py で生成されたトークンを検証
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # 有効期限のチェック (PyJWTが自動でも行いますが、念のためログ出力用に確認)
        exp = payload.get("exp")
        if exp:
            now = datetime.now(timezone.utc).timestamp()
            if now > exp:
                logger.warning(f"WebSocket auth failed: Token expired. (exp: {exp}, now: {now})")
                return False
        
        return True

    except jwt.ExpiredSignatureError:
        logger.warning("WebSocket auth failed: Signature has expired.")
        return False
    except jwt.InvalidTokenError as e:
        # 署名不一致の場合はここでエラーになります
        logger.warning(f"WebSocket auth failed: Invalid token. Error: {e}")
        return False
    except Exception as e:
        logger.error(f"WebSocket auth system error: {e}", exc_info=True)
        return False