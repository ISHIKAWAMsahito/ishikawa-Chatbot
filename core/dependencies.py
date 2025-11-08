from fastapi import Request,HTTPException
from core.config import SUPER_ADMIN_EMAILS, ALLOWED_CLIENT_EMAILS

# --- 認証関数 (Auth0用) ---
# --- 管理者用認証 ---
def require_auth(request: Request):
    """管理者用認証 (SUPER_ADMIN_EMAILS のみ許可)"""
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=307, headers={'Location': '/login'})
    
    # 念のため小文字に変換して比較
    user_email = user.get('email', '').lower()
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]

    if user_email in super_admin_emails_lower:
        return user
    else:
        # スーパー管理者リストに含まれていない場合は、管理者ページへのアクセスを拒否
        raise HTTPException(status_code=403, detail="管理者ページへのアクセス権がありません。")

# --- 学生用認証 ---
def require_auth_client(request: Request):
    """クライアント用認証 (ALLOWED_CLIENT_EMAILS または SUPER_ADMIN_EMAILS を許可)"""
    user = request.session.get('user')

    # 1. 最初に、ログインしているかどうかを確認します。
    if not user:
        raise HTTPException(status_code=307, headers={'Location': '/login'})

    # 2. ユーザー情報があることが確定してから、メールアドレスを取得します。
    user_email = user.get('email', '').lower()

    # 3. 比較対象の許可リストを両方とも小文字に変換しておきます。
    allowed_emails_lower = [email.lower() for email in ALLOWED_CLIENT_EMAILS]
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]

    # --- デバッグ用のprint文（安全な場所に移動） ---
    # print("--- クライアント認証チェック ---")
    # print(f"ログイン試行中のメアド (小文字化後): '[{user_email}]'")
    # print(f"クライアント許可リスト: {allowed_emails_lower}")
    # print(f"管理者許可リスト: {super_admin_emails_lower}")
    # print(f"クライアントリストに含まれているか？: {user_email in allowed_emails_lower}")
    # print(f"管理者リストに含まれているか？: {user_email in super_admin_emails_lower}")
    # print("--------------------")
    # -----------------------------------------------------------

    # 4. 認証チェックを実行します。
    # (クライアント許可リスト、または管理者許可リストのどちらかに含まれていればOK)
    if (user_email in allowed_emails_lower or
        user_email in super_admin_emails_lower):
        return user
    else:
        raise HTTPException(status_code=403, detail="このサービスへのアクセスは許可されていません。")
