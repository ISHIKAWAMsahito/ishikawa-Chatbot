from fastapi.testclient import TestClient
from main import app  # main.pyからFastAPIアプリケーションをインポート

client = TestClient(app)

def test_health_check():
    """
    /health エンドポイントが正常に動作するかをテスト
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_serve_client_html():
    """
    ルートURL (/) にアクセスした際に、正常にHTMLが返されるかをテスト
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers['content-type']

def test_admin_redirect_for_unauthenticated_user():
    """
    未認証のユーザーが /admin にアクセスした際に、
    ログインページへリダイレクトされるかをテスト
    """
    response = client.get("/admin", follow_redirects=False) # 自動リダイレクトを無効化
    assert response.status_code == 307 # 307 Temporary Redirect
    assert "/login" in response.headers.get("location", "")

