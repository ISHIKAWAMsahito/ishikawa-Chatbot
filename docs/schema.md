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

## RPC Functions (Stored Procedures)

### `match_documents`
ベクトル検索用関数。
- Arguments: `p_query_embedding`, `p_match_count`, `p_collection_name`...
- Returns: `table (id, content, metadata, similarity)`

### `match_documents_hybrid`
キーワード検索とベクトル検索のハイブリッド。