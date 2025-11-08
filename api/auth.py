import os
import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, FileResponse, Response
from core.config import oauth, AUTH0_DOMAIN, AUTH0_CLIENT_ID, SUPER_ADMIN_EMAILS, ALLOWED_CLIENT_EMAILS, BASE_DIR
from core.dependencies import require_auth, require_auth_client

router = APIRouter()

# 認証エンドポイント
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
    request.session.pop('user', None)
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]):
        return RedirectResponse(url='/')
    return RedirectResponse(f"https://{AUTH0_DOMAIN}/v2/logout?returnTo={request.url_for('serve_client')}&client_id={AUTH0_CLIENT_ID}")

# HTMLファイル提供
@router.get("/", response_class=FileResponse)
async def serve_client(request: Request, user: dict = Depends(require_auth_client)):
    return FileResponse(os.path.join(BASE_DIR, "static", "client.html"))

@router.get("/admin", response_class=FileResponse)
async def serve_admin(request: Request, user: dict = Depends(require_auth)):
    return FileResponse(os.path.join(BASE_DIR, "static", "admin.html"))

@router.get("/DB.html", response_class=FileResponse)
async def serve_db_page(request: Request, user: dict = Depends(require_auth)):
    db_path = os.path.join(BASE_DIR, "static", "DB.html")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="DB.html not found")
    return FileResponse(db_path)

@router.get("/feedback-stats", response_class=FileResponse)
async def serve_feedback_stats(request: Request, user: dict = Depends(require_auth)):
    return FileResponse(os.path.join(BASE_DIR, "static", "feedback_stats.html"))

@router.get("/style.css", response_class=FileResponse)
async def serve_css():
    return FileResponse(os.path.join(BASE_DIR, "static", "style.css"))

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)