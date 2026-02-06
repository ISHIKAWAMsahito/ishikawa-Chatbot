# Database Schema (Supabase)

## Tables

### `documents`
RAG用のドキュメントチャンクを保存するテーブル。
- `id`: uuid (PK)
- `content`: text (子チャンクの内容)
- `metadata`: jsonb (親チャンクの内容 `parent_content` や `source` を含む)
- `embedding`: vector(1536) (OpenAI embeddings)
- `created_at`: timestamptz

### `users` (Auth)
- Supabase Authと連携

### `anonymous_comments`
管理者ダッシュボードの統計データ (`/api/admin/stats/data`) やユーザーフィードバックに使用。
- `id`: bigint (PK)
- `content`: text (ユーザーからのフィードバックやコメント内容)
- `created_at`: timestamptz (並び替えに使用)

### `qa_fallbacks` (推定)
`api/fallbacks` ルーターで使用される、Q&Aの修正や追加データを管理するテーブル。
- `id`: bigint (PK)
- `question`: text
- `answer`: text
- `embedding`: vector(1536) (検索用)
- `created_at`: timestamptz

## RPC Functions (Stored Procedures)

### `match_documents`
ベクトル検索用関数。
- Arguments: `p_query_embedding`, `p_match_count`, `p_collection_name`...
- Returns: `table (id, content, metadata, similarity)`

### `match_documents_hybrid`
キーワード検索とベクトル検索のハイブリッド。
- Arguments: `p_query_text`, `p_query_embedding`, `p_match_count`, `p_collection_name`
- Returns: `table (id, content, metadata, similarity)`

### `match_fallback_qa`
フォールバックQ&A（`qa_fallbacks`）を検索する関数。
- Arguments: `p_query_embedding`, `p_match_count`
- Returns: `table (id, question, answer, similarity)`