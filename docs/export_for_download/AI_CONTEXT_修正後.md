プロジェクトコンテキスト
FastAPI, Supabase, LangChainを採用したRAGチャットシステム。 「Parent-Child Chunking」を用いた高精度な検索と、管理者向け機能を提供する。

コーディング指針（High Resolution）
- **セキュリティ**: 指針「5. セキュリティ」および「6. 禁止事項」のセキュリティ関連項目を必ず守ること。違反すると認可 bypass・セッション偽造・SSRF・XSS・情報漏洩などの脆弱性につながる。詳細は `docs/SECURITY_VULNERABILITY_REPORT.md` を参照すること。
1. データモデルと型安全性 (Strict Typing)
Dict禁止: APIの入出力や内部データの受け渡しには、生辞書 (dict) ではなく、必ず Pydantic の BaseModel を定義・使用すること。

CamelCase変換: APIレスポンスはフロントエンドに合わせてキャメルケース (camelCase) に自動変換する設定を入れる。

Type Hints: 関数引数と戻り値には必ず型ヒントを付ける。Any の使用は極力避け、ジェネリクスやUnion型で具体化すること。

2. エラーハンドリングとログ (Robustness)
防御的プログラミング: エラー時は必ずスタックトレースを含めてログ出力する。

logging.error(f"Error in function_name: {e}", exc_info=True) を徹底すること。

起動時バリデーション (Fail Fast):

アプリケーション起動時 (lifespan) に、必須の環境変数や SettingsManager などのコアコンポーネントが初期化されているか厳格にチェックする。

欠落や初期化失敗がある場合は、ValueError を発生させて即座にプロセスを停止 (Fail Fast) させること。

## 3. インターフェースの整合性と最適化ロジックの遵守 (Interface & Logic Integrity)
- **最適化パイプラインの遵守:**
    - `search.py` における「ハイブリッド検索 → リランク (上位5件精査/6.0点以上) → LitM配置 (U字型) → 多様性フィルタ (70%重複カット)」というパイプラインは、応答時間18秒と高精度を両立するための**絶対的な標準仕様**である。
    - 修正時、これらのステップを簡略化したり、パラメータ（閾値等）を独断で変更したりしてはならない。
- **使用モデルの厳格な固定 (Model Standardization):**
    - **LLM (生成・リランク):** 高速処理と精度を両立するため、すべての生成タスク（チャット回答、リランク判定、クエリ拡張など）において、原則として **`models/gemini-2.5-flash`** を使用すること。コード内で `gemini-1.5-flash` や `gemini-pro` がハードコードされている場合は、直ちに `2.5` 系へ修正する。
    - **Embedding (埋め込み):** 安定動作を保証するため、必ず **`models/gemini-embedding-001`** を使用すること。`text-embedding-004` 等の新しいモデルは、現在のAPIバージョンでは 404 エラー（Not Found）を引き起こすため使用を禁止する。
- **Embeddingモデルの固定:**
    - 埋め込み生成（Embedding）には、安定動作を保証するため必ず `models/gemini-embedding-001` を使用すること。
    - `text-embedding-004` 等の新しいモデルは、現在のAPIバージョンでは 404 エラー（Not Found）を引き起こすため使用を禁止する。
- **依存関係の全量チェック:**
    - メソッドを呼び出す際は、必ず呼び出し先（`llm.py` 等）にその定義が存在し、引数が合致しているか確認すること。
- **コード提供の原則 :**
    - ファイル更新時は、指示があるまで**修正箇所、変更箇所のコード（ブロック）**を提供すること。
- **変更時のダブルチェック:**
    - 呼び出し元（`chat_logic.py`）を直したら、必ず呼び出し先（`search.py` や `llm.py`）の整合性を確認し、両方の整合が取れたコードを提示すること。

1. 設定管理 (Configuration Management)
環境変数のエイリアス対応: Render等のPaaS環境では、予約語回避のために変数名が変更される場合がある。core.config 内で必ずエイリアス処理を行うこと。

例: SECRET_KEY = os.getenv("APP_SECRET_KEY") or os.getenv("SECRET_KEY")。

機密情報の秘匿: ログ出力時にAPIキーやシークレットキーを表示しないようマスキング（末尾数桁のみ表示など）すること。

---

## 5. セキュリティ (Security) — 脆弱性の高いコードを生成しない

以下のルールを守ること。違反すると認可 bypass・セッション偽造・SSRF・XSS・情報漏洩などの重大な脆弱性につながる。

### 5.1 認可・認証 (Authorization)
- **管理者向けAPI**: ドキュメント一覧・取得、設定取得・変更、コレクション操作、フィードバック分析・統計など、管理者のみが触るべきエンドポイントには **必ず `Depends(require_auth)` を付与する**。認証なしでデータ取得・設定変更を公開しない。
- **一般向けAPI**: 学生・ログイン済みユーザー向けの投稿・履歴取得などは、必要に応じて `Depends(require_auth_client)` でログイン必須とする。
- **公開してよいもの**: ヘルスチェック（`/health`, `/healthz`）、フロント初期化用の公開設定（Supabase URL/anon key など）のみ認証なしでよい。それ以外は役割に応じて認可を付与する。

### 5.2 セッション・秘密鍵 (Secrets)
- **本番での秘密鍵必須**: セッション用の秘密鍵（APP_SECRET_KEY / SECRET_KEY）は、本番環境（例: RENDER 設定時）では **未設定なら起動を失敗させる**（Fail Fast）。デフォルトの `"default-insecure-key"` を本番で使ってはならない。
- **参照元**: 秘密鍵は main.py やルーターで `os.getenv` を直接読まず、**core.config で定義した定数**（SECRET_KEY, APP_SECRET_KEY）を使用する。

### 5.3 リダイレクト・Host ヘッダー (Open Redirect 対策)
- **Host ヘッダーを信頼しない**: ログイン後リダイレクト先などを組み立てる際、`request.headers.get("host")` をそのまま使わない。**許可ホストのリスト（ALLOWED_HOSTS）** で検証し、リストにない場合は `request.url.netloc` など信頼できる値にフォールバックすること。
- **core.config**: ALLOWED_HOSTS は環境変数で設定し、Render デプロイ時はサービスホスト名を追加する。

### 5.4 外部通信・スクレイピング (TLS・SSRF)
- **TLS検証を無効にしない**: 外部への HTTP/HTTPS リクエスト（スクレイピング等）で **`verify=False` を使ってはならない**。デフォルトの `verify=True` を維持する。
- **URL検証**: ユーザー入力の URL で外部取得する場合は **スキームを https（必要なら http）に限定**し、**localhost・127.0.0.1・プライベートIP・メタデータ用ホスト（169.254.x.x 等）は拒否**する（SSRF 対策）。許可リストまたは拒否リストでホストを検証すること。

### 5.5 エラー応答・情報開示
- **詳細をクライアントに返さない**: `HTTPException(status_code=500, detail=str(e))` や `detail=f"...{str(e)}"` のように、**例外メッセージやスタックトレースをそのまま detail に返してはならない**。クライアントには汎用メッセージ（例: 「処理に失敗しました。」）のみ返し、**詳細は `logging.error(..., exc_info=True)` でログにのみ記録**する。

### 5.6 入力検証・サニタイズ
- **長さ制限**: チャットの質問文・検索クエリ・コメントなど、ユーザー入力には **Pydantic の `max_length` や `min_length` で妥当な上限・下限を設定**する。極端に長い入力による DoS や API コスト暴騰を防ぐ。
- **検索パラメータ**: DB の `ilike` 等に渡す検索文字列は **ワイルドカード（%, _）やカンマの除去・エスケープ** を行う。未サニタイズのままクエリに埋め込まない。
- **URL・パス**: 検索・スクレイピングに使う URL は上記 5.4 の検証のほか、パス traversal を防ぐ必要がある場合はパス部分も検証する。

### 5.7 フロントエンド・XSS
- **innerHTML に未エスケープのユーザー・LLM 由来文字列を代入しない**: 表示用に **エスケープ関数（escapeHtml）** を用意し、ユーザー入力や API/LLM から返った文字列はエスケープしてから `innerHTML` に渡す。`[text](url)` 形式のリンクだけを安全に `<a>` に変換する場合は、**リンクテキストと URL の両方をエスケープ**し、URL は **https/http のみ許可**する。
- **画像・リソースURL**: `<img src="...">` などに挿入する URL は **スキームを https（または http）に限定**し、`javascript:` 等を拒否する。
- **Markdown 表示**: `marked` 等で HTML 化したあと `innerHTML` に渡す場合は、**script タグや on* 属性を除去**するなど、出力をサニタイズしてから表示する。

---

## 6. 禁止事項 (Anti-Patterns)
❌ os.getenv の直接呼び出し: main.py やルーター内で os.getenv を直接使用しない。必ず core.config で定義された定数を使用すること。

❌ print() デバッグ: 運用ログの可読性を保つため、すべて logging を使用する。

❌ 同期的なファイル/ネットワーク操作: async def 内で標準の open() や requests を使用しないこと。

❌ **認証なしの管理者API**: ドキュメント取得・設定変更・統計取得など管理者向けエンドポイントに require_auth を付けない実装を追加しないこと。

❌ **本番でデフォルト秘密鍵**: 本番環境で APP_SECRET_KEY 未設定のまま "default-insecure-key" で起動する実装にしないこと。

❌ **Host ヘッダーをそのままリダイレクトに使用**: リダイレクトURIを Host ヘッダーのみで組み立てないこと。ALLOWED_HOSTS で検証すること。

❌ **verify=False**: 外部 HTTP クライアントで TLS 検証を無効にしないこと。

❌ **HTTPException(detail=str(e))**: 500 系エラーで例外メッセージをそのままクライアントに返さないこと。

❌ **未エスケープの innerHTML**: ユーザー・LLM 由来の文字列をエスケープせずに innerHTML に代入しないこと。

---

## 7. テスト (Testing Strategy)
テストフレームワークは pytest を使用する。

実際のSupabaseや外部APIには接続せず、unittest.mock でモック化すること。
