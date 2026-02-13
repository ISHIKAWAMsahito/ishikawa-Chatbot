# 🎓 学生支援チャットボットシステム

Retrieval-Augmented Generation (RAG) を活用した、学生支援チャットボットです。
無料のクラウドサービスを組み合わせ、学生でも持続的に運用可能な構成を目指しました。

## 🚀 主な機能

- **🤖 AI質問応答**: 自然言語で大学関連の質問に回答
- **📚 公式情報ベースの回答**: 大学サイトや資料を参照し、ハルシネーションを抑制
- **🔐 セキュアなアクセス制御**: 大学ドメインと開発者限定の認証
- **⚙️ リアルタイム設定反映**: 管理者の設定変更が即時に学生画面へ反映

## 🏗️ システム構成

- **フロントエンド層**: HTML/CSS + JavaScript (WebSocket)
- **アプリケーション層**: FastAPI + Gemini API + Auth0
- **データベース層**: Supabase (PostgreSQL + pgvector)
- **インフラ・監視層**: Render + Docker

## 📂 プロジェクト構成

FastAPIのベストプラクティスに基づき、機能ごとにモジュール分割されています。

```text
.
├── main.py                  # アプリケーション本体、Lifespan管理、ルーター登録
├── static/                  # 静的ファイル (フロントエンド)
│   ├── client.html          # 学生用チャット画面
│   ├── admin.html           # 管理者用ダッシュボード
│   ├── DB.html              # ナレッジベース(DB)管理画面
│   ├── stats.html           # フィードバック統計・分析画面
│   └── style.css            # 共通スタイルシート
│
├── api/                     # APIエンドポイント (ルーター)
│   ├── auth.py              # Auth0認証、HTML配信
│   ├── chat.py              # チャット履歴取得・クライアント用エンドポイント
│   ├── documents.py         # ナレッジ登録・検索・管理
│   ├── feedback.py          # ユーザーフィードバック受信
│   ├── system.py            # ヘルスチェック、設定管理
│   └── stats.py             # 管理者用統計データ提供
│
├── services/                # ビジネスロジック
│   ├── chat_logic.py        # チャットフロー制御、履歴管理
│   ├── search.py            # 検索ロジック (クエリ拡張・リランク)
│   ├── llm.py               # Gemini API通信・リトライ処理
│   ├── document_processor.py # テキスト抽出・チャンキング
│   ├── storage.py           # Supabase Storage連携
│   └── feedback.py          # フィードバック集計ロジック
│
├── core/                    # 中核設定・コンポーネント
│   ├── config.py            # 環境変数・定数定義
│   ├── database.py          # DB接続 (Supabase) 管理
│   ├── dependencies.py      # 認証依存関係 (require_auth)
│   ├── ws_auth.py           # WebSocket認証
│   └── settings.py          # 動的設定管理 (SettingsManager)
│
└── models/                  # データモデル
    └── schemas.py           # Pydanticモデル (リクエスト/レスポンス定義)

⚙️ 技術スタック
フロントエンド: HTML5 + Vanilla JS (WebSocket対応)

バックエンド: Python (FastAPI)

AI/LLM: Google Gemini API (gemini-2.5-flash, gemini-embedding-001)

検索/RAG: ハイブリッド検索, クエリ拡張, Rerank

DB: Supabase (PostgreSQL + pgvector)

認証: Auth0 (OAuth2.0)

インフラ: Render (Web Service), Docker

## 🛠️ セットアップ

### 1. 環境変数の設定
プロジェクトルートに `.env` ファイルを作成し、以下の変数を設定してください：

```properties
# --- 基本設定 & セキュリティ ---
APP_SECRET_KEY=your_random_secret_key
JWT_SECRET_KEY=your_jwt_secret_key

# --- 管理者 & アクセス制御 ---
ADMIN_USERNAME=admin_user
ADMIN_PASSWORD_HASH=hashed_password_string
SUPER_ADMIN_EMAILS=admin@example.com,dev@example.com
ALLOWED_CLIENT_EMAILS=student@univ.ac.jp  # 許可するメールドメイン等
CLIENT_EMAILS=test_student@univ.ac.jp     # テスト用など特定の許可メアド

# --- 認証 (Auth0) ---
AUTH0_DOMAIN=your_domain.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret

# --- AI & LLM (Gemini) ---
GEMINI_API_KEY=your_gemini_api_key

# --- データベース (Supabase) ---
USE_SUPABASE=true
SUPABASE_URL=[https://xxx.supabase.co](https://xxx.supabase.co)
SUPABASE_SERVICE_KEY=your_service_key     # role: service_role (サーバーサイド用)
SUPABASE_ANON_KEY=your_anon_key           # role: anon (クライアントサイド用)

# --- 監視 & トレース (LangSmith) ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=[https://api.smith.langchain.com](https://api.smith.langchain.com)
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name

ローカル開発 (Docker)
docker run -it --rm `
  -p 8000:8000 `
  -e PORT=8000 `
  -e ENVIRONMENT=local `
  --env-file "C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env" `
  my-fastapi-app

デプロイ (Render)
リポジトリ内の render.yaml を使用してデプロイ可能です。
services:
  - type: web
    name: fastapi-chatbot
    runtime: python
    plan: free

🔒 セキュリティ
認証: Auth0を利用したメール全体をチェックした制限

ロール管理:

学生: client.html のみアクセス可

管理者: admin.html, DB.html, stats.html へのアクセス権限

データ保護: 個人情報（氏名・学籍番号等）の入力禁止運用

📈 今後の改善予定
音声入力インターフェースの実装

多言語対応
