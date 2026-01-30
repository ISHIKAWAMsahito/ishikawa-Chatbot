🎓 学生支援チャットボットシステム
Retrieval-Augmented Generation (RAG) を活用した、札幌学院大学向けの学生支援チャットボットです。
無料のクラウドサービスを組み合わせ、学生でも持続的に運用可能な構成を目指しました。

🚀 主な機能
- 🤖 AI質問応答：自然言語で大学関連の質問に回答
- 📚 公式情報ベースの回答：大学サイトや資料を参照
- 🔐 セキュアなアクセス制御：大学ドメインと開発者限定
- ⚙️ リアルタイム設定反映：管理者の変更が即時に学生画面へ反映

🏗️ システム構成
フロントエンド層: client.html / admin.html / DB.html
アプリケーション層: FastAPI + Gemini API + Auth0
データベース層: Supabase (PostgreSQL + pgvector)
インフラ・監視層: Render + Docker



📂 プロジェクト構成
├── main.py              # メインアプリケーション
├── client.html          # 学生用画面
├── admin.html           # 管理者用画面
├── DB.html              # DB管理画面
├── requirements.txt     # 依存ライブラリ
├── Dockerfile           # Docker設定
├── docker-compose.yml   # ローカル開発用
├── render.yaml          # Renderデプロイ設定
├── prometheus.yml       # Prometheus設定
├── data/                # 設定・ログ保存
│   ├── shared_settings.json
│   └── feedback.json
└── README.md



⚙️ 技術スタック
- フロントエンド: HTML + JavaScript (WebSocket対応)
- バックエンド: FastAPI
- AI: Gemini API (gemini-embedding-001, gemini-2.5-flash)
- DB: Supabase (PostgreSQL + pgvector)
- 認証: Auth0 (OAuth2.0)
- インフラ: Render (無料プラン), Docker
- 監視: Uptime Robot

🛠️ セットアップ
1. 環境変数を設定
.env ファイルを作成し、以下を記入：
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=your_service_key
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
AUTH0_DOMAIN=your_domain.auth0.com
APP_SECRET_KEY=your_random_secret_key


2. ローカル開発
docker-compose up -d


- アプリ: http://localhost:8000
3. Render でデプロイ
# render.yaml を利用
services:
  - type: web
    name: fastapi-chatbot
    runtime: python
    plan: free



🔒 セキュリティ
- 学生: client.html
- 管理者: admin.html / DB.html / stats.html
- 個人情報入力は禁止（氏名・学籍番号など）

📈 今後の改善予定
- 音声入力・多言語対応
- LINE連携
