# =========================================================
# 学生画面用 設定配布API
# =========================================================
@app.get("/api/client/config")
def get_client_config():
    """学生画面へSupabase接続情報を渡す"""
    if not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Server Config Error: ANON KEY not found")

    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY
    }