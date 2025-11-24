import os
import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, FileResponse, Response
from core.config import oauth, AUTH0_DOMAIN, AUTH0_CLIENT_ID, SUPER_ADMIN_EMAILS, ALLOWED_CLIENT_EMAILS, BASE_DIR

router = APIRouter()

# ---------------------------------------------------------
# 認証エンドポイント
# ---------------------------------------------------------
@router.get('/login')
async def login_auth0(request: Request):
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    return await oauth.auth0.authorize_redirect(request, request.url_for('auth'))

@router.get('/auth')
async def auth(request: Request):
    """Auth0からのコールバックを処理"""
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

        # 管理者は /admin にリダイレクト
        if user_email in super_admin_emails_lower:
            return RedirectResponse(url='/admin')
        # クライアントは / にリダイレクト
        elif user_email in allowed_emails_lower:
            return RedirectResponse(url='/')
        # 許可されていないユーザーはログアウト
        else:
            logging.warning(f"Unauthorized login attempt by: {user_email}")
            return RedirectResponse(url='/logout')
            
    logging.error("Failed to get userinfo from Auth0.")
    return RedirectResponse(url='/login')

@router.get('/logout')
async def logout(request: Request):
    """ログアウト処理 - セッションクリアと Auth0 リダイレクト"""
    request.session.pop('user', None)
    
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]):
        logging.warning("Auth0 configuration missing")
        return RedirectResponse(url='/', status_code=302)
    
    # 現在のホストを取得（http/https を自動判定）
    scheme = request.url.scheme
    host = request.url.netloc
    return_to = f"{scheme}://{host}/login"
    
    logout_url = (
        f"https://{AUTH0_DOMAIN}/v2/logout?"
        f"client_id={AUTH0_CLIENT_ID}&"
        f"returnTo={return_to}"
    )
    
    logging.info(f"User logged out. Redirecting to: {logout_url}")
    return RedirectResponse(url=logout_url, status_code=302)

# ---------------------------------------------------------
# HTMLファイル提供
# ---------------------------------------------------------

@router.get("/", response_class=FileResponse)
async def serve_client(request: Request):
    """クライアント画面"""
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
    """管理者画面 - セッション確認と権限チェック"""
    user = request.session.get('user')
    
    # セッションがない場合はログイン画面へ
    if not user:
        logging.info("No session found, redirecting to login")
        return RedirectResponse(url='/login', status_code=302)
    
    user_email = user.get('email', '').lower()
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]
    
    # 管理者でない場合はホーム画面へ
    if user_email not in super_admin_emails_lower:
        logging.warning(f"Non-admin user attempted to access /admin: {user_email}")
        return RedirectResponse(url='/', status_code=302)

    # 管理者の場合は admin.html を返す
    admin_path = os.path.join(BASE_DIR, "static", "admin.html")
    if not os.path.exists(admin_path):
        raise HTTPException(status_code=404, detail="admin.html not found")
    
    logging.info(f"Admin user {user_email} accessed admin panel")
    return FileResponse(admin_path)

@router.get("/DB.html", response_class=FileResponse)
async def serve_db_page(request: Request):
    """DB管理画面 - セッション確認と権限チェック"""
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
    
    logging.info(f"Admin user {user_email} accessed DB management")
    return FileResponse(db_path)

@router.get("/feedback-stats", response_class=FileResponse)
async def serve_feedback_stats(request: Request):
    """フィードバック統計画面 - セッション確認"""
    user = request.session.get('user')
    if not user:
        return RedirectResponse(url='/login', status_code=302)
    
    stats_path = os.path.join(BASE_DIR, "static", "feedback_stats.html")
    if not os.path.exists(stats_path):
        raise HTTPException(status_code=404, detail="feedback_stats.html not found")
    
    return FileResponse(stats_path)

@router.get("/style.css", response_class=FileResponse)
async def serve_css():
    """CSS提供"""
    css_path = os.path.join(BASE_DIR, "static", "style.css")
    if not os.path.exists(css_path):
        raise HTTPException(status_code=404, detail="style.css not found")
    return FileResponse(css_path)

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Favicon"""
    return Response(status_code=204)

@router.get("/debug/session")
async def debug_session(request: Request):
    """デバッグエンドポイント - セッション状態確認（開発用）"""
    user = request.session.get('user')
    user_email = user.get('email') if user else None
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS] if SUPER_ADMIN_EMAILS else []
    
    return {
        "session_exists": user is not None,
        "user_email": user_email,
        "is_admin": user_email.lower() in super_admin_emails_lower if user_email else False,
        "admin_emails": super_admin_emails_lower
    }