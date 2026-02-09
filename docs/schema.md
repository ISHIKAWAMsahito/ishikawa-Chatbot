# Database Schema (Supabase)

## Tables

### `documents`
RAG用のドキュメントチャンクを保存するテーブル。
- `id`: uuid (PK)
- `content`: text (子チャンクの内容)
- `metadata`: jsonb (親チャンクの内容 `parent_content` や `source` を含む)
- `embedding`: vector(1536) (Gemini embeddings)
- `created_at`: timestamptz

### `users` (Auth)
- Supabase Authと連携

### `anonymous_comments`
管理者ダッシュボードの統計データ (`/api/admin/stats/data`) やユーザーフィードバックに使用。
- `id`: uuid (PK) - 以前はbigintだったがUUIDに変更
- `content`: text (ユーザーからのフィードバックやコメント内容。APIレスポンスでは `comment` として扱われる)
- `created_at`: timestamptz (並び替えに使用)
- `rating`: text (現状APIでは未使用だが、将来的に 'good'/'bad' 等を格納)

### `category_fallbacks`
`api/fallbacks` ルーターで使用される、カテゴリ別のQ&A修正や追加データを管理するテーブル。
- `id`: bigint (PK)
- `category_name`: text (カテゴリ名。例: 'campus_life', 'tuition')
- `question`: text
- `answer`: text
- `embedding`: vector(1536) (Gemini Embedding)
    - **Note**: Python API側では文字列として取得される場合があるため、`json.loads` によるパース変換が必須。
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