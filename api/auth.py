import os
import logging
from urllib.parse import quote_plus  # ★追加: URLエンコード用
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, FileResponse, Response
from core.config import (
    oauth,
    AUTH0_DOMAIN,
    AUTH0_CLIENT_ID,
    SUPER_ADMIN_EMAILS,
    ALLOWED_CLIENT_EMAILS,
    BASE_DIR,
    ALLOWED_HOSTS,
)

router = APIRouter()
# #システムの入り口となる認証と、HTMLファイルの配信を担当しています。

# Auth0連携: /login でAuth0へリダイレクトし、/auth でコールバックを受け取ります。

# 権限管理:

# SUPER_ADMIN_EMAILS: 管理者（admin.htmlなどへのアクセス権）

# ALLOWED_CLIENT_EMAILS: 一般ユーザー（学生など、client.htmlへのアクセス権）

# これらリストに含まれないメールアドレスはログアウトさせられます。

# 静的ファイル配信: 認証状態に基づいて、SPA（Single Page Application）のHTMLファイル（client.html, admin.html, DB.html 等）を直接返しています。

# =========================================================
#  URL生成ヘルパー (Host ヘッダー検証: オープンリダイレクト対策)
# =========================================================
def get_safe_redirect_uri(request: Request, path: str) -> str:
    host_header = (request.headers.get("host") or "").strip()
    host_for_check = host_header.split(":")[0].lower() if host_header else ""
    if host_for_check not in ALLOWED_HOSTS:
        logging.warning(f"Rejected Host header for redirect: {host_header!r}. Using request netloc.")
        host_header = request.url.netloc or "localhost"
    scheme = "https" if ("onrender.com" in host_header or "render.com" in host_header) else request.url.scheme
    clean_path = path if path.startswith("/") else f"/{path}"
    return f"{scheme}://{host_header}{clean_path}"

# ---------------------------------------------------------
# 認証エンドポイント
# ---------------------------------------------------------
@router.get('/login')
async def login_auth0(request: Request):
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    
    redirect_uri = get_safe_redirect_uri(request, '/auth')
    return await oauth.auth0.authorize_redirect(request, redirect_uri)

@router.get('/auth')
async def auth(request: Request):
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    
    try:
        token = await oauth.auth0.authorize_access_token(request)
    except Exception as e:
        logging.error(f"Auth0 access token error: {e}")
        return RedirectResponse(url='/login')

    if userinfo := token.get('userinfo'):
        request.session['user'] = dict(userinfo)
        user_email = userinfo.get('email', '').lower()
        
        super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]
        allowed_emails_lower = [email.lower() for email in ALLOWED_CLIENT_EMAILS]

        if user_email in super_admin_emails_lower:
            return RedirectResponse(url='/admin')
        elif user_email in allowed_emails_lower:
            return RedirectResponse(url='/')
        else:
            logging.warning(f"Unauthorized login attempt by: {user_email}")
            return RedirectResponse(url='/logout')
            
    return RedirectResponse(url='/login')

@router.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]):
        return RedirectResponse(url='/', status_code=302)
    
    # 戻り先URLを作成 (https://.../login)
    return_to = get_safe_redirect_uri(request, '/login')
    
    # ★修正: URLエンコードを行う (:// などを %3A%2F%2F に変換)
    # これでAuth0が確実にURLを認識できるようになります
    encoded_return_to = quote_plus(return_to)
    
    logout_url = (
        f"https://{AUTH0_DOMAIN}/v2/logout?"
        f"client_id={AUTH0_CLIENT_ID}&"
        f"returnTo={encoded_return_to}"
    )
    
    return RedirectResponse(url=logout_url, status_code=302)

# ... (HTML配信部分などは変更なし。そのまま残してください) ...
# ---------------------------------------------------------
# ページ配信
# ---------------------------------------------------------
@router.get("/", response_class=FileResponse)
async def serve_client(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/login', status_code=302)
    client_path = os.path.join(BASE_DIR, "static", "client.html")
    if not os.path.exists(client_path):
        raise HTTPException(status_code=404, detail="client.html not found")
    return FileResponse(client_path)

@router.get("/admin")
@router.get("/admin.html")
async def serve_admin(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/login', status_code=302)
    user_email = user.get('email', '').lower()
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]
    if user_email not in super_admin_emails_lower:
        return RedirectResponse(url='/', status_code=302)
    admin_path = os.path.join(BASE_DIR, "static", "admin.html")
    if not os.path.exists(admin_path):
        raise HTTPException(status_code=404, detail="admin.html not found")
    return FileResponse(admin_path)

@router.get("/DB.html", response_class=FileResponse)
async def serve_db_page(request: Request):
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/login', status_code=302)
    user_email = user.get('email', '').lower()
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]
    if user_email not in super_admin_emails_lower:
        return RedirectResponse(url='/', status_code=302)
    db_path = os.path.join(BASE_DIR, "static", "DB.html")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="DB.html not found")
    return FileResponse(db_path)

@router.get("/feedback-stats", response_class=FileResponse)
async def serve_feedback_stats(request: Request):
    # 1. ログインチェック
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/login', status_code=302)
    
    # 2. ★追加: SUPER_ADMIN_EMAILS かどうかのチェック
    user_email = user.get('email', '').lower()
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]
    
    # リストに入っていない人がアクセスしたらトップページへ飛ばす
    if user_email not in super_admin_emails_lower:
        return RedirectResponse(url='/', status_code=302)
    
    # 3. ファイルの配信
    stats_path = os.path.join(BASE_DIR, "static", "feedback_stats.html")
    if not os.path.exists(stats_path):
        raise HTTPException(status_code=404, detail="feedback_stats.html not found")
    return FileResponse(stats_path)

@router.get("/style.css", response_class=FileResponse)
async def serve_css():
    css_path = os.path.join(BASE_DIR, "static", "style.css")
    if not os.path.exists(css_path):
        raise HTTPException(status_code=404, detail="style.css not found")
    return FileResponse(css_path)

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)