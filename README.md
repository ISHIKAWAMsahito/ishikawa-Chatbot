🎓 学生支援チャットボットシステム
Retrieval-Augmented Generation (RAG) を活用した、学生支援チャットボットプラットフォームです。
単なる問い合わせの自動化に留まらず、現場主導で学生支援体制を改善し続けるための基盤として設計されています。

※本システムは研究・試行的に開発しているものであり、学内公式ではありません。

🚀 主な機能
🤖 AI質問応答 (RAG): 
　大学サイトやマニュアル等の公式資料に基づき、自然言語で学生の質問に回答します。


🧠 AI業務改善アドバイザー : 
　学生とAIの過去の対話ログを分析し、職員に対して「学生の悩み」の抽出や「窓口対応の改善案」を提案します 。
+4


📊 統計・分析ダッシュボード:
　フィードバックの統計グラフと、AIとの改善相談チャットを統合した管理者用インターフェースを提供します 。

🔐 セキュアなアクセス制御: 
　Auth0連携により、特定のユーザーのみにアクセスを制限します。

⚙️ 現場主導のナレッジ管理:
　技術者に頼らず、職員自身がRAG対象ドキュメントを更新・修正できる設計にしています。

🏗️ システム構成

フロントエンド層: 
　HTML / Vanilla JS / Chart.js / WebSocket (リアルタイム設定反映) 


アプリケーション層: 
　FastAPI (Python) / Gemini API / Auth0 (OAuth2.0) 

データベース層:
　Supabase (PostgreSQL + pgvector)

インフラ・監視層: 
　Render / Docker / LangSmith (トレース・評価)

📂 プロジェクト構成
.
├── main.py                  # アプリケーション本体、Lifespan管理、ルーター登録
├── static/                  # 静的ファイル (フロントエンド)
│   ├── client.html          # 学生用チャット画面
│   ├── admin.html           # 管理者用ダッシュボード
│   ├── DB.html              # ナレッジベース(DB)管理画面
│   ├── stats.html           # 統計・AI改善相談画面 (Update) [cite: 42]
│   └── style.css            # 共通スタイルシート
│
├── api/                     # APIエンドポイント (ルーター) 
│   ├── auth.py              # Auth0認証処理
│   ├── chat.py              # チャット・履歴・フィードバック API
│   ├── documents.py         # ナレッジ(RAG用)登録・検索 API
│   ├── system.py            # 設定管理・ヘルスチェック API
│   └── stats.py             # 統計・対話ログ分析 API (Update)
│
├── services/                # ビジネスロジック 
│   ├── chat_logic.py        # RAGパイプライン制御・ログ保存トリガー (Update)
│   ├── chat_log.py          # 対話ログのDB保存・永続化処理 (New)
│   ├── search.py            # クエリ拡張・リランク・ハイブリッド検索
│   ├── llm.py               # Gemini API連携・リトライ処理
│   ├── prompts.py           # プロンプト一元管理 (Update)
│   └── document_processor.py # テキスト抽出・チャンキング
│
├── core/                    # 中核設定
│   ├── database.py          # Supabase(PostgreSQL)接続管理
│   ├── config.py            # 環境変数・定数定義
│   └── dependencies.py      # 認証・認可依存関係 (require_auth)
│
└── models/                  # データモデル
    └── schemas.py           # Pydanticモデル定義 (Update)

🛠️ セットアップ
1. 環境変数の設定
.env ファイルに以下の設定が必要です。
# --- 基本設定 & セキュリティ ---
APP_SECRET_KEY=...
# --- 管理者設定 ---
SUPER_ADMIN_EMAILS=admin@example.com
# --- AI & LLM (Gemini) ---
GEMINI_API_KEY=...
# --- データベース (Supabase) ---
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
# --- 監視 (LangSmith) ---
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...

2. ローカル開発 (Docker)
docker build -t chatbot-app .
docker run -p 8000:8000 --env-file .env chatbot-app

🔒 セキュリティと運用ビジョン

自律的な改善サイクル:
　日常的な運用の中で現場職員が「回答の不十分さ」に気づいた際、即座に知識ベースを修正できる体制を支援します。


対話ログの資産化:
　学生とのやり取りを蓄積し、システム自身が改善案を提示する「相談パートナー」へと進化させることで、教職員の支援態勢アップデートを加速させます 。

📈 今後の展望
音声入力インターフェース