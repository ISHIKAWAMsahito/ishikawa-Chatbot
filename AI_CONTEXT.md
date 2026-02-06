# プロジェクトコンテキスト
FastAPI, Supabase, LangChainを採用したRAGチャットシステム。
「Parent-Child Chunking」を用いた高精度な検索と、管理者向け機能を提供する。

# コーディング指針（High Resolution）

## 1. データモデルと型安全性 (Strict Typing)
- **Dict禁止:** APIの入出力や内部データの受け渡しには、生辞書 (`dict`) ではなく、必ず Pydantic の `BaseModel` を定義・使用すること。
- **CamelCase変換:** APIレスポンスはフロントエンド（JS/TS）に合わせてキャメルケース (`camelCase`) に自動変換する設定を入れること。
    - `ConfigDict(alias_generator=to_camel, populate_by_name=True)` を使用。
- **Type Hints:** 関数引数と戻り値には必ず型ヒントを付ける。`Any` の使用は極力避け、ジェネリクスやUnion型で具体化する。

## 2. エラーハンドリングとログ (Robustness)
- **防御的プログラミング:** `main.py` の `logging` 設定に従い、エラー時は必ずスタックトレースを含めてログ出力する。
    - `logging.error(f"Error in function_name: {e}", exc_info=True)`
- **HTTP例外:** APIエンドポイント内で例外をキャッチした場合、そのまま 500 を返さず、適切な `HTTPException` に変換する。
    - リソース不足: 404 Not Found
    - 権限不足: 403 Forbidden
    - バリデーションエラー: 400 Bad Request
    - 外部サービス(Supabase)ダウン: 503 Service Unavailable

## 3. データベース操作 (Supabase Best Practices)
- **RPC優先:** 複雑なクエリやベクトル検索は、Pythonコードでフィルタリングせず、PostgreSQL側の関数 (RPC) を作成・呼び出しするよう提案すること。
- **コネクション管理:** `core.database.db_client` シングルトンを必ず使用し、リクエストごとに `create_client` を行わない。
- **セッション管理:** DB接続エラー時のリトライ処理を考慮する。

## 4. RAG / ドキュメント処理 (Critical Logic)
- **親子関係の維持:** `SimpleDocumentProcessor` を修正・拡張する際は、`parent_splitter` (1500文字) と `child_splitter` (400文字) の比率を勝手に変更しないこと。
- **メタデータ必須:** ドキュメント登録時、`metadata` には必ず `parent_content`, `source`, `page` を含めること。

## 5. 禁止事項 (Anti-Patterns)
- ❌ `os.getenv` を各ファイルで直接呼び出すこと（`core.config` を経由せよ）。
- ❌ `print()` デバッグを残すこと。
- ❌ `document_processor.py` 以外の場所で PDF/Docx 解析ロジックを書くこと（ロジックの散逸防止）。
- ❌ 非同期関数 (`async def`) の中でブロッキングなI/O操作（標準の `open()` や `requests`）を行うこと。

## 6. テスト (Testing Strategy)
- テストフレームワークは `pytest` を使用。
- `unittest.mock` を使い、`SupabaseClientManager` の `client` をモック化すること。実際のSupabaseには接続しないテストを書くこと。