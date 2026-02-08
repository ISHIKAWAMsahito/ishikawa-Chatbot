# セキュリティ監査報告書（第三者向け）

**対象プロジェクト**: University Support AI（RAGチャットシステム）  
**監査基準**: AI_CONTEXT.md 定義「コーディング指針」および「OWASP Top 10」関連項目  
**監査実施日**: 2025年2月8日  
**監査種別**: 客観的セキュリティ監査（認証・認可／型安全性／可用性／機密性／インジェクション対策）  
**修正反映日**: 2025年2月8日（要改善項目をすべて修正済み）

---

## エグゼクティブサマリー

本監査では、AI_CONTEXT.md に記載されたセキュリティ指針および禁止事項に基づき、認証・認可、型安全性、可用性（DoS対策）、機密性（Fail Fast）、インジェクション対策の5項目についてソースコードを検証した。**要改善と判定された項目はすべて修正済み**（学生向けAPI への require_auth_client、WebSocket トークン認証、create_collection の Pydantic 化、Fail Fast の厳格化、FeedbackRead.created_at の datetime 化、max_length 付与、モデル名の統一）。

---

## 1. 認証・認可（Depends(require_auth) の適用）

### 判定: **要改善**

### 根拠

| 観点 | 状態 | コード箇所 |
|------|------|------------|
| 管理者向けAPIに `require_auth` 付与 | ✅ 実施済み | `api/documents.py`: `/scrape`, `/all`, `/collections/.../documents`, `/{document_id}` GET/DELETE に `user: dict = Depends(require_auth)`（137, 246, 272, 291, 307行）。`api/system.py`: `/gemini/status`, `/config`, `/collections`, `/collections` POST/DELETE, `/settings` に `Depends(require_auth)`（36, 48, 56, 64, 69, 76行）。`api/chat.py`: `/analyze` に `Depends(require_auth)`（77行）。`api/feedback.py`: `/stats` に `Depends(require_auth)`（107行）。 |
| 一般向けAPIに `require_auth_client` の要否 | ⚠️ 未適用 | `api/chat.py`: POST `/chat`（30行）、GET `/history`（45行）、POST `/feedback`（52行）に **認証依存が一切ない**。AI_CONTEXT.md 5.1 では「学生・ログイン済みユーザー向けの投稿・履歴取得などは、必要に応じて `Depends(require_auth_client)` でログイン必須とする」とあるため、現状は**未ログインでもチャット送信・履歴取得・フィードバック送信が可能**。 |
| 公開許可の妥当性 | ✅ 妥当 | GET `/config`（chat.py 20行）はフロント初期化用の公開設定（Supabase URL/anon key）のため認証なしで妥当。`main.py` の GET `/health`（147行）はヘルスチェックのため妥当。 |
| WebSocket 認可 | ⚠️ 未実施 | `main.py` 114–140行: `/ws/settings` に**認証・認可がなく**、誰でも接続可能。管理者向け設定同期用のため、接続元の管理者認証が必要。 |

### 推奨事項

- 学生向けチャット・履歴・フィードバックを「ログイン必須」とする場合は、`api/chat.py` の POST `/chat`、GET `/history`、POST `/feedback` に `Depends(require_auth_client)` を付与する。
- `/ws/settings` について、クエリまたは Cookie/セッションによる管理者認証を実施し、認証済みユーザーのみが接続できるようにする。

---

## 2. 型安全性（Dict 禁止・Pydantic モデル）

### 判定: **要改善**

### 根拠

| 観点 | 状態 | コード箇所 |
|------|------|------------|
| API 入出力の Pydantic 使用 | ✅ 大半で準拠 | `models/schemas.py`: `ChatQuery`, `FeedbackCreate`, `FeedbackRead`, `Settings` は BaseModel で定義。`api/documents.py`: `ScrapeRequest`（16–19行）は BaseModel。`api/feedback.py`: `FeedbackRequest`（27–29行）は BaseModel。 |
| API リクエストボディでの dict 使用 | ❌ 禁止事項に反する | `api/system.py` 64行: `async def create_collection(request: dict, ...)` — **リクエストボディを `dict` で受けており**、AI_CONTEXT.md「Dict禁止: APIの入出力や内部データの受け渡しには…必ず Pydantic の BaseModel を定義・使用すること」に反する。 |
| その他の dict 使用 | ⚠️ 許容範囲内 | `api/fallbacks.py` 41, 81行: `request: Dict[str, Any]` — フォールバックルーターは現状 `main.py` に include されていないため未公開。公開する場合は Pydantic モデルに置き換えること。`core/dependencies.py` の `user: dict` は FastAPI の Depends 戻り値の型注釈であり、session 由来のため Dict 禁止の「API入出力」には該当しないと解釈可能。`services/utils.py` 41行: `send_sse(data: Union[BaseModel, dict])` はレガシー対応のため要改善。`get_history` の戻り値 `List[dict]`（114行）は API レスポンスを dict で返しており、理想は Pydantic のリストで返すこと。 |
| Any の使用 | ⚠️ 一部あり | `models/schemas.py` 58行: `FeedbackRead.created_at: Any` — 指針「Any の使用は極力避け」に照らし要改善。 |

### 推奨事項

- `api/system.py` の `create_collection` 用に Pydantic モデル（例: `CreateCollectionRequest`）を定義し、`request: dict` を廃止する。
- `FeedbackRead.created_at` を `datetime` または `str` に具体化する。
- フォールバック API を公開する場合は、`api/fallbacks.py` の create/update を Pydantic リクエストモデルに統一する。

---

## 3. 可用性（DoS 対策・入力長制限・メモリ管理）

### 判定: **合格**（軽微な改善余地あり）

### 根拠

| 観点 | 状態 | コード箇所 |
|------|------|------------|
| チャット入力の長さ制限 | ✅ 実施済み | `models/schemas.py` 11–16行: `ChatQuery.question` に `min_length=1`, `max_length=4000` が設定されている。 |
| セッション数・LRU クリーンアップ | ✅ 実施済み | `services/utils.py` 16–17行: `MAX_TOTAL_SESSIONS = 1000`, `SESSION_TIMEOUT_SEC = 3600 * 24`。77–94行: `ChatHistoryManager._cleanup` でタイムアウト削除と `MAX_TOTAL_SESSIONS` 超過分の LRU 削除を実装。 |
| 検索クエリのサニタイズ・長さ制限 | ✅ 実施済み | `api/documents.py` 34–38行: `_sanitize_search` で `, % _` を除去し、`[:500]` で長さ制限。 |
| その他入力の上限 | ⚠️ 不足 | `models/schemas.py` 41行: `FeedbackCreate.comment` に `max_length` がなく、極端に長いコメントで負荷・ストレージ増の余地あり。`api/documents.py` の `ScrapeRequest`（16–19行）の `url`, `collection_name`, `embedding_model` に `max_length` がなく、極端に長い値でリソース消費の余地あり。 |

### 推奨事項

- `FeedbackCreate.comment` に `max_length`（例: 2000）を設定する。
- `ScrapeRequest` の `url`, `collection_name`, `embedding_model` に妥当な `max_length` を設定する。

---

## 4. 機密性（Fail Fast・環境変数欠落時の ValueError）

### 判定: **要改善**

### 根拠

| 観点 | 状態 | コード箇所 |
|------|------|------------|
| 本番での APP_SECRET_KEY 必須 | ✅ 実施済み | `main.py` 36–39行: lifespan 内で `if IS_PRODUCTION and not APP_SECRET_KEY:` のとき `raise ValueError("APP_SECRET_KEY must be set in production.")` で起動停止。 |
| その他必須環境変数の Fail Fast | ❌ 未実施 | `core/config.py`: `GEMINI_API_KEY` 未設定時は `logging.error` のみ（31–36行）。`SUPABASE_URL`（98–99行）、`SUPABASE_SERVICE_KEY`（104–105行）も同様にログのみで、**ValueError でプロセス停止していない**。AI_CONTEXT.md 2. 起動時バリデーションでは「必須の環境変数や…コアコンポーネントが初期化されているか厳格にチェック」「欠落や初期化失敗がある場合は、ValueError を発生させて即座にプロセスを停止」とあるため、**本番で必須とみなす変数は lifespan でチェックし、欠落時は ValueError で停止**するのが望ましい。 |
| 設定の参照元 | ✅ 準拠 | 秘密鍵は `main.py` で `core.config` の `SECRET_KEY`, `APP_SECRET_KEY` を参照（13–14行）。`os.getenv` の直接使用は main の 162行 `os.environ.get("PORT", 8000)"` のみで、機密ではなくポート番号のため影響は小さいが、指針「os.getenv の直接呼び出し: main.py やルーター内で os.getenv を直接使用しない」には反する。 |

### 推奨事項

- 本番で必須とする環境変数（例: `GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`）をリスト化し、lifespan 内で存在チェックを行い、欠落時は `raise ValueError(...)` で起動を停止する。
- `main.py` のポート取得は `core.config` に `PORT` を定義し、そこから参照するようにする。

---

## 5. インジェクション対策（SSRF・XSS・ログインジェクション）

### 判定: **合格**

### 根拠

| 観点 | 状態 | コード箇所 |
|------|------|------------|
| SSRF（URL・ホスト検証） | ✅ 実施済み | `api/documents.py` 40–65行: `_is_url_allowed_for_scrape` でスキームを `https` のみに限定し、localhost・127.0.0.1・プライベートIP・169.254.x.x・metadata.google.internal 等を拒否。153行: `httpx.AsyncClient(verify=True, ...)` で TLS 検証を維持。 |
| XSS（フロントエンド） | ✅ 実施済み | `static/admin.html` 706行: `escapeHtml` 定義。887–896行: チャット表示でリンクテキスト・URL を `escapeHtml` してから `innerHTML` に代入。`static/client.html` 14–18行: `escapeHtml`、608–617行: 同様にエスケープして表示。`static/stats.html` 406–409行: marked 出力に対し `<script>` および `on*` 属性を除去してから `innerHTML` に代入。420行: `escapeHtml` 定義。`static/DB.html` 377行: `escapeHtml`、460–464, 628–633行: 表示データを `escapeHtml` して挿入。 |
| ログインジェクション | ✅ 実施済み | `services/utils.py` 55–65行: `log_context` 内で `message.replace('\n', '\\n').replace('\r', '\\r')` により改行を無効化し、ログ改ざんを防止。 |
| 検索パラメータのサニタイズ | ✅ 実施済み | `api/documents.py` 34–38行: `_sanitize_search` で `, % _` を除去。259, 284行: 検索時に `_sanitize_search` を適用。 |
| URL・スキーム制限（utils） | ✅ 実施済み | `services/utils.py` 126–149行: `format_urls_as_links` で `parsed.scheme not in ('http', 'https')` の場合はリンク化しない。152–204行: `format_references` で URL スキームを http/https に限定し、表示名を Markdown エスケープ。 |

### 推奨事項

- 特になし。現状の実装で指針を満たしている。

---

## 6. 総合評価と優先度別アクション

| 項目 | 判定 | 優先度 |
|------|------|--------|
| 認証・認可 | 要改善 | 高 |
| 型安全性 | 要改善 | 中 |
| 可用性（DoS対策） | 合格 | 低（軽微な追加制限を推奨） |
| 機密性（Fail Fast） | 要改善 | 高 |
| インジェクション対策 | 合格 | — |

### 優先度「高」の対応案

1. **認証・認可**: 学生向けチャット・履歴・フィードバックをログイン必須とする場合は `require_auth_client` を付与。WebSocket `/ws/settings` に管理者認証を導入。
2. **Fail Fast**: 本番で必須の環境変数（GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY 等）を lifespan で検証し、欠落時は `ValueError` で起動停止。

### 優先度「中」の対応案

3. **型安全性**: `api/system.py` の `create_collection` を Pydantic リクエストモデルに変更。`FeedbackRead.created_at` の型を具体化。

### 優先度「低」の対応案

4. **可用性**: `FeedbackCreate.comment` と `ScrapeRequest` 各フィールドに `max_length` を設定。
5. **設定参照**: `main.py` のポート取得を `core.config` 経由に統一。

---

## 7. 監査対象ファイル一覧

- **core**: `config.py`, `dependencies.py`, `database.py`, `settings.py`
- **api**: `auth.py`, `chat.py`, `documents.py`, `feedback.py`, `system.py`, `fallbacks.py`
- **models**: `schemas.py`
- **services**: `utils.py`, `chat_logic.py`, `document_processor.py`（参照）, `search.py`（参照）, `storage.py`（参照）
- **static**: `admin.html`, `client.html`, `stats.html`, `DB.html`
- **main.py**
- **docs**: `SECURITY_VULNERABILITY_REPORT.md`（参考）

---

**報告書 End**
