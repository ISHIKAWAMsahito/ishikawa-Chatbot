# プロジェクトコンテキスト
FastAPI, Supabase, LangChainを採用したRAGチャットシステム。 「Parent-Child Chunking」を用いた高精度の検索と、管理者向け機能を提供する。

# コーディング指針（High Resolution）

## 1. データモデルと型安全性 (Strict Typing)
- **Dict禁止**: APIの入出力や内部データの受け渡しには、生辞書 (dict) ではなく、必ず Pydantic の BaseModel を定義・使用すること。
- **CamelCase変換**: APIレスポンスはフロントエンドに合わせてキャメルケース (camelCase) に自動変換する設定を入れる。
- **Type Hints**: 関数引数と戻り値には必ず型ヒントを付ける。Any の使用は極力避け、ジェネリクスやUnion型で具体化すること。
- **Supabase Vector型ハンドリング (重要)**: 
    - Supabase (pgvector) の `vector` カラムは、ライブラリによって `List[float]` ではなく **`str` (例: `"[0.01, -0.02...]"`)** として返される場合がある。
    - APIレスポンスとして返す前（Pydanticモデルに渡す前）に、必ず **`json.loads()` 等でパースして `List[float]` に変換** する処理を挟むこと。
    - **禁止事項**: DBから取得した `embedding` データをそのまま Pydantic モデルに渡してはならない（バリデーションエラーの原因となる）。

## 2. エラーハンドリングとログ (Robustness)
- **防御的プログラミング**: エラー時は必ずスタックトレースを含めてログ出力する。
  `logging.error(f"Error in function_name: {e}", exc_info=True)` を徹底すること。
- **起動時バリデーション (Fail Fast)**:
  アプリケーション起動時 (lifespan) に、必須の環境変数や SettingsManager などのコアコンポーネントが初期化されているか厳格にチェックする。
  欠落や初期化失敗がある場合は、`ValueError` を発生させて即座にプロセスを停止 (Fail Fast) させること。

## 3. インターフェース設計
- **RESTful**: エンドポイントはリソース指向で設計する。
- **認証**: 
    - 管理者機能は `Depends(require_auth)` で保護する。
    - 学生向け機能は `Depends(require_auth_client)` で保護する。

---

# 禁止事項（Critical）

### 8.1 認証・認可
- **学生向けAPIの認証なし**: チャット送信・履歴取得・フィードバック送信など、ログイン済みユーザー向けエンドポイントには **必ず `Depends(require_auth_client)` を付与**する。認証なしのまま追加しない。
- **管理者向けWebSocketの認証なし**: 管理者専用の WebSocket（例: `/ws/settings`）には **接続前に短期トークン（GET /api/admin/system/ws-token）を取得し、`?token=xxx` で接続**する。トークン検証なしで接続を許可しない。
- **管理者向けAPIの認証漏れ**: ドキュメント一覧・設定取得・統計・分析など管理者向けエンドポイントには **必ず `Depends(require_auth)` を付与**する。

### 8.2 型安全性（Dict 禁止）
- **API リクエストボディで `request: dict` を使わない**: すべて **Pydantic の BaseModel を定義し、その型で受け取る**。例: `create_collection(request: CreateCollectionRequest)`。
- **レスポンスや内部データの `Any` を避ける**: `created_at: Any` などは **`str` や `datetime` に具体化**する。
- **API 入出力に生の `Dict[str, Any]` を使わない**: 作成・更新のリクエストは必ず Pydantic モデル（例: `FallbackCreate`, `FallbackUpdate`）で受け取る。

### 8.3 起動時バリデーション（Fail Fast）
- **本番で必須の環境変数をチェックしない**: 本番（RENDER 等）では **`GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `APP_SECRET_KEY`** が未設定の場合、サーバーを起動させてはならない。