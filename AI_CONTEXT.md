プロジェクトコンテキスト
FastAPI, Supabase, LangChainを採用したRAGチャットシステム。 「Parent-Child Chunking」を用いた高精度な検索と、管理者向け機能を提供する。

コーディング指針（High Resolution）
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

## 3. インターフェースの整合性と更新ルール (Interface Integrity)
- **依存関係の全量チェック:** - 関数やメソッドを呼び出す際は、必ず呼び出し先（`llm.py` や `search.py` 等）にそのメソッドが定義されているか、引数の型が合っているかを確認すること。
    - 呼び出し先を修正する際は、呼び出し元（`chat_logic.py` 等）に影響がないか、既存のメソッドを削除・改名していないかを必ず確認すること。
- **全量コード提供の原則 (No Omissions):**
    - ファイルを更新する場合、既存の重要な関数（例: `analyze_feedback_trends` や `get_db`）を「省略」として削らないこと。
    - インポートエラーを防ぐため、常にそのファイルで必要なすべてのインポート文と関数を揃えた「完全なコード」を生成すること。
- **循環参照の回避:**
    - `api` -> `services` -> `core` の階層構造を守り、逆方向のインポート（例: `core` が `api` を呼ぶ）が発生しないよう設計すること。
- **変更時のダブルチェック:** - ファイルA（呼び出し元）を修正したら、必ずファイルB（呼び出し先）の整合性を確認し、必要なら両方の修正コードを提示すること。片方だけの修正で済ませない。
4. 設定管理 (Configuration Management)
環境変数のエイリアス対応: Render等のPaaS環境では、予約語回避のために変数名が変更される場合がある。core.config 内で必ずエイリアス処理を行うこと。

例: SECRET_KEY = os.getenv("APP_SECRET_KEY") or os.getenv("SECRET_KEY")。

機密情報の秘匿: ログ出力時にAPIキーやシークレットキーを表示しないようマスキング（末尾数桁のみ表示など）すること。

5. 禁止事項 (Anti-Patterns)
❌ os.getenv の直接呼び出し: main.py やルーター内で os.getenv を直接使用しない。必ず core.config で定義された定数を使用すること。

❌ print() デバッグ: 運用ログの可読性を保つため、すべて logging を使用する。

❌ 同期的なファイル/ネットワーク操作: async def 内で標準の open() や requests を使用しないこと。

6. テスト (Testing Strategy)
テストフレームワークは pytest を使用する。

実際のSupabaseや外部APIには接続せず、unittest.mock でモック化すること。