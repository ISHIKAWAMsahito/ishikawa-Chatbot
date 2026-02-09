import jwt
import logging
from datetime import datetime, timezone
from core.config import SECRET_KEY

logger = logging.getLogger(__name__)

# アルゴリズムは HS256 に固定
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
        return False
    
    try:
        # core.config.SECRET_KEY を使用してデコード
        # verify_exp=True はデフォルトで有効ですが、念のため明示
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # 期限切れチェック (pyjwtが自動で行いますが、念のためのロジック)
        exp = payload.get("exp")
        if exp:
            now = datetime.now(timezone.utc).timestamp()
            if now > exp:
                logger.warning("WebSocket token expired (manual check).")
                return False
                
        return True

    except jwt.ExpiredSignatureError:
        logger.warning("WebSocket token has expired.")
        return False
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid WebSocket token: {e}")
        return False
    except Exception as e:
        logger.error(f"WebSocket token validation error: {e}")
        return False