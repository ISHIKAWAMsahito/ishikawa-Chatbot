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

## 3. インターフェースの整合性と最適化ロジックの遵守 (Interface & Logic Integrity)
- **最適化パイプラインの遵守:**
    - `search.py` における「ハイブリッド検索 → リランク (上位5件精査/6.0点以上) → LitM配置 (U字型) → 多様性フィルタ (70%重複カット)」というパイプラインは、応答時間18秒と高精度を両立するための**絶対的な標準仕様**である。
    - 修正時、これらのステップを簡略化したり、パラメータ（閾値等）を独断で変更したりしてはならない。
- **依存関係の全量チェック:**
    - メソッドを呼び出す際は、必ず呼び出し先（`llm.py` 等）にその定義が存在し、引数が合致しているか確認すること。
- **全量コード提供の原則 (No Omissions):**
    - ファイル更新時は、常にそのファイル内の**すべてのインポート文、クラス、メソッドを揃えた「完全なコード」**を提供すること。一部の関数を `...` で省略することは、デプロイ時の ImportError や依存性の欠落を招くため厳禁とする。
- **変更時のダブルチェック:**
    - 呼び出し元（`chat_logic.py`）を直したら、必ず呼び出し先（`search.py` や `llm.py`）の整合性を確認し、両方の整合が取れたコードを提示すること。
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