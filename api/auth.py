import os
import logging
from urllib.parse import quote_plus
from fastapi import APIRouter, Request, HTTPException
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

# ---------------------------------------------------------
# URL生成ヘルパー (HTTPS強制 & Host検証)
# ---------------------------------------------------------
def get_safe_redirect_uri(request: Request, path: str) -> str:
    # 1. Hostヘッダーの検証
    host_header = (request.headers.get("host") or "").strip()
    host_for_check = host_header.split(":")[0].lower() if host_header else ""
    
    if host_for_check not in ALLOWED_HOSTS:
        logging.warning(f"Rejected Host header: {host_header}. Using default.")
        # ALLOWED_HOSTSの先頭(本番ならRenderドメイン、開発ならlocalhost)を使用
        host_header = ALLOWED_HOSTS[0] if ALLOWED_HOSTS else "localhost"

    # 2. Schemeの決定 (Render対応)
    # X-Forwarded-Proto が https なら https を採用
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto == "https" or "onrender.com" in host_header:
        scheme = "https"
    else:
        scheme = request.url.scheme

    clean_path = path if path.startswith("/") else f"/{path}"
    return f"{scheme}://{host_header}{clean_path}"

# ---------------------------------------------------------
# 認証エンドポイント
# ---------------------------------------------------------
@router.get('/login')
async def login_auth0(request: Request):
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    
    # HTTPS化された正しいコールバックURLを生成
    redirect_uri = get_safe_redirect_uri(request, '/auth')
    return await oauth.auth0.authorize_redirect(request, redirect_uri)

@router.get('/auth')
async def auth(request: Request):
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    
    try:
        # stateチェックなどを含むトークン取得
        token = await oauth.auth0.authorize_access_token(request)
    except Exception as e:
        logging.error(f"Auth0 access token error: {e}")
        # エラー時はログイン画面へ戻す
        return RedirectResponse(url='/login')

    if userinfo := token.get('userinfo'):
        request.session['user'] = dict(userinfo)
        user_email = userinfo.get('email', '').lower()
        
        super_admins = [e.lower() for e in SUPER_ADMIN_EMAILS]
        allowed_clients = [e.lower() for e in ALLOWED_CLIENT_EMAILS]

        if user_email in super_admins:
            return RedirectResponse(url='/admin')
        elif user_email in allowed_clients:
            return RedirectResponse(url='/')
        else:
            logging.warning(f"Unauthorized login attempt: {user_email}")
            return RedirectResponse(url='/logout')
            
    return RedirectResponse(url='/login')

@router.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]):
        return RedirectResponse(url='/', status_code=302)
    
    return_to = get_safe_redirect_uri(request, '/login')
    encoded_return_to = quote_plus(return_to)
    
    logout_url = (
        f"https://{AUTH0_DOMAIN}/v2/logout?"
        f"client_id={AUTH0_CLIENT_ID}&"
        f"returnTo={encoded_return_to}"
    )
    return RedirectResponse(url=logout_url, status_code=302)

# ---------------------------------------------------------
# ページ配信 (変更なし)
# ---------------------------------------------------------
@router.get("/", response_class=FileResponse)
async def serve_client(request: Request):
    user = request.session.get('user')
    if not user: return RedirectResponse(url='/login', status_code=302)
    path = os.path.join(BASE_DIR, "static", "client.html")
    return FileResponse(path) if os.path.exists(path) else Response(status_code=404)

@router.get("/admin", response_class=FileResponse)
@router.get("/admin.html", response_class=FileResponse)
async def serve_admin(request: Request):
    user = request.session.get('user')
    if not user: return RedirectResponse(url='/login', status_code=302)
    if user.get('email', '').lower() not in [e.lower() for e in SUPER_ADMIN_EMAILS]:
        return RedirectResponse(url='/', status_code=302)
    path = os.path.join(BASE_DIR, "static", "admin.html")
    return FileResponse(path) if os.path.exists(path) else Response(status_code=404)

@router.get("/DB.html", response_class=FileResponse)
async def serve_db(request: Request):
    user = request.session.get('user')
    if not user: return RedirectResponse(url='/login', status_code=302)
    if user.get('email', '').lower() not in [e.lower() for e in SUPER_ADMIN_EMAILS]:
        return RedirectResponse(url='/', status_code=302)
    path = os.path.join(BASE_DIR, "static", "DB.html")
    return FileResponse(path) if os.path.exists(path) else Response(status_code=404)

@router.get("/style.css", response_class=FileResponse)
async def serve_css():
    path = os.path.join(BASE_DIR, "static", "style.css")
    return FileResponse(path) if os.path.exists(path) else Response(status_code=404)

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)