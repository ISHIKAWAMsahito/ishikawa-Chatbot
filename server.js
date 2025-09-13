// 必要なライブラリを読み込む
const express = require('express');
const mysql = require('mysql2/promise');
const cors = require('cors'); // CORSを許可するためのライブラリ
const axios = require('axios'); // HTTP リクエスト用

// Expressアプリを作成
const app = express();
app.use(express.json()); // JSON形式のリクエストを扱えるようにする
app.use(cors()); // すべてのオリジンからのリクエストを許可（開発用）

// --------------------------------------------------
// ▼▼▼ あなたのMySQLデータベース情報に書き換えてください ▼▼▼
const dbConfig = {
    host: 'localhost',              // データベースサーバーのホスト名 or IPアドレス
    user: 'root',                   // MySQLのユーザー名
    password: 'ishikawa3150',      // MySQLのパスワード
    database: 'chatbot_db'  // 使用するデータベース名
};
// ▲▲▲ ここまで ▲▲▲
// --------------------------------------------------

// Ollama設定
const OLLAMA_BASE_URL = 'http://localhost:11434';
const DEFAULT_MODEL = 'llama3.2';

// MySQL接続プールを作成（パフォーマンス向上）
const pool = mysql.createPool({
    ...dbConfig,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

// ユーティリティ関数：Ollama接続確認
async function checkOllamaConnection() {
    try {
        const response = await axios.get(`${OLLAMA_BASE_URL}/api/tags`);
        return response.status === 200;
    } catch (error) {
        console.error('Ollama接続エラー:', error.message);
        return false;
    }
}

// ユーティリティ関数：MySQL接続確認
async function checkMySQLConnection() {
    try {
        const connection = await pool.getConnection();
        await connection.ping();
        connection.release();
        return true;
    } catch (error) {
        console.error('MySQL接続エラー:', error.message);
        return false;
    }
}

// ユーティリティ関数：データベース検索
async function searchDatabase(query, limit = 10) {
    try {
        const connection = await pool.getConnection();
        
        // （変更なし）
      const searchQueries = [
            {
                table: 'products',
                columns: ['id', 'name', 'description', 'price'],
                searchColumn: 'name'
            },
            {
                table: 'products',
                columns: ['id', 'name', 'description', 'price'],
                searchColumn: 'description'
            }
        ];

        const results = [];
        const searchTerm = `%${query}%`;

        for (const searchQuery of searchQueries) {
            // ▼▼▼ ここから変更 ▼▼▼
            // LIMIT句を直接文字列に埋め込む
            const sql = `SELECT ${searchQuery.columns.join(', ')} FROM ${searchQuery.table} WHERE ${searchQuery.searchColumn} LIKE ? LIMIT ${Number(limit)}`;
            
            // executeに渡す引数をsearchTermのみにする
            const [rows] = await connection.execute(sql, [searchTerm]);
            // ▲▲▲ ここまで変更 ▲▲▲
            
            if (rows.length > 0) {
                results.push({
                    table: searchQuery.table,
                    search_field: searchQuery.searchColumn,
                    results: rows
                });
            }
        }

        connection.release();
        return results;
    } catch (error) {
        console.error('データベース検索エラー:', error);
        return [];
    }
}

// ユーティリティ関数：プロンプト生成
function generatePrompt(userMessage, dbResults) {
    let prompt = `あなたは親切なAIアシスタントです。以下のデータベース検索結果を参考に、ユーザーの質問に答えてください。

ユーザーの質問: ${userMessage}

`;

    if (dbResults && dbResults.length > 0) {
        prompt += `データベース検索結果:\n`;
        dbResults.forEach((result, index) => {
            prompt += `\n${index + 1}. テーブル: ${result.table} (検索フィールド: ${result.search_field})\n`;
            prompt += `結果: ${JSON.stringify(result.results, null, 2)}\n`;
        });
        prompt += `\n上記のデータベース情報を参考に、適切で有用な回答を提供してください。`;
    } else {
        prompt += `データベースからは関連する情報が見つかりませんでしたが、一般的な知識で回答してください。`;
    }

    return prompt;
}

// APIエンドポイント：ヘルスチェック
app.get('/health', async (req, res) => {
    const ollamaStatus = await checkOllamaConnection();
    const mysqlStatus = await checkMySQLConnection();
    
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        services: {
            ollama: ollamaStatus,
            mysql: mysqlStatus,
            rag: ollamaStatus && mysqlStatus
        }
    });
});

// APIエンドポイント：利用可能なモデル一覧
app.get('/models', async (req, res) => {
    try {
        const response = await axios.get(`${OLLAMA_BASE_URL}/api/tags`);
        res.json(response.data);
    } catch (error) {
        console.error('モデル取得エラー:', error.message);
        res.status(503).json({ error: 'Ollamaサーバーに接続できません' });
    }
});

// APIエンドポイント：データベース検索のみ
app.post('/search', async (req, res) => {
    const { query, limit = 10 } = req.body;

    if (!query) {
        return res.status(400).json({ error: 'クエリが空です' });
    }

    try {
        const results = await searchDatabase(query, limit);
        res.json({
            query: query,
            results: results,
            count: results.reduce((sum, r) => sum + r.results.length, 0)
        });
    } catch (error) {
        console.error('検索エラー:', error);
        res.status(500).json({ error: 'データベース検索でエラーが発生しました' });
    }
});

// APIエンドポイント：チャット（RAG機能付き）
app.post('/chat', async (req, res) => {
    const { 
        message, 
        model = DEFAULT_MODEL, 
        temperature = 0.7, 
        max_tokens = 1000,
        use_rag = true 
    } = req.body;

    if (!message) {
        return res.status(400).json({ error: 'メッセージが空です' });
    }

    try {
        let dbResults = [];
        let prompt = message;

        // RAG検索を実行
        if (use_rag) {
            dbResults = await searchDatabase(message, 5);
            prompt = generatePrompt(message, dbResults);
        }

        // Ollamaに送信
        const ollamaResponse = await axios.post(`${OLLAMA_BASE_URL}/api/generate`, {
            model: model,
            prompt: prompt,
            temperature: temperature,
            max_tokens: max_tokens,
            stream: false
        });

        // レスポンスを整形
        const aiResponse = ollamaResponse.data.response;
        
        res.json({
            reply: aiResponse,
            data: dbResults,
            metadata: {
                model: model,
                temperature: temperature,
                max_tokens: max_tokens,
                db_results_count: dbResults.reduce((sum, r) => sum + r.results.length, 0),
                rag_used: use_rag
            }
        });

    } catch (error) {
        console.error('チャットエラー:', error.message);
        
        // エラーの種類に応じた応答
        if (error.response?.status === 404) {
            res.status(404).json({ error: `モデル '${model}' が見つかりません` });
        } else if (error.code === 'ECONNREFUSED') {
            res.status(503).json({ error: 'Ollamaサーバーに接続できません' });
        } else {
            res.status(500).json({ error: 'サーバー内部でエラーが発生しました' });
        }
    }
});

// APIエンドポイント：Ollamaプロキシ（直接Ollama APIを叩く）
app.post('/ollama/generate', async (req, res) => {
    try {
        const response = await axios.post(`${OLLAMA_BASE_URL}/api/generate`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Ollama APIエラー:', error.message);
        res.status(503).json({ error: 'Ollama APIでエラーが発生しました' });
    }
});

// APIエンドポイント：データベース情報取得
app.get('/database/info', async (req, res) => {
    try {
        const connection = await pool.getConnection();
        
        // テーブル一覧を取得
        const [tables] = await connection.execute('SHOW TABLES');
        const tableInfo = [];
        
        for (const table of tables) {
            const tableName = Object.values(table)[0];
            const [columns] = await connection.execute(`DESCRIBE ${tableName}`);
            const [count] = await connection.execute(`SELECT COUNT(*) as count FROM ${tableName}`);
            
            tableInfo.push({
                name: tableName,
                columns: columns.map(col => ({
                    name: col.Field,
                    type: col.Type,
                    nullable: col.Null === 'YES'
                })),
                row_count: count[0].count
            });
        }
        
        connection.release();
        
        res.json({
            database: dbConfig.database,
            tables: tableInfo
        });
    } catch (error) {
        console.error('データベース情報取得エラー:', error);
        res.status(500).json({ error: 'データベース情報の取得に失敗しました' });
    }
});

// エラーハンドリング
app.use((err, req, res, next) => {
    console.error('予期しないエラー:', err);
    res.status(500).json({ error: '予期しないエラーが発生しました' });
});

// サーバーを起動するポート番号
const PORT = process.env.PORT || 3001;

app.listen(PORT, async () => {
    // ▼▼▼ この行が目印です ▼▼▼
    console.log("★★★★★ 目印付きの新しいサーバーが起動しました ★★★★★"); 
    // ▲▲▲ この行が目印です ▲▲▲

    console.log(`🚀 サーバーがポート${PORT}で起動しました。 http://localhost:${PORT}`);
    
    // 起動時にサービス状態をチェック
    const ollamaStatus = await checkOllamaConnection();
    const mysqlStatus = await checkMySQLConnection();
    
    console.log('📊 サービス状態:');
    console.log(`  Ollama: ${ollamaStatus ? '✅ 接続済み' : '❌ 未接続'}`);
    console.log(`  MySQL: ${mysqlStatus ? '✅ 接続済み' : '❌ 未接続'}`);
    console.log(`  RAG機能: ${ollamaStatus && mysqlStatus ? '✅ 利用可能' : '❌ 未対応'}`);
    
    if (!ollamaStatus) {
        console.log('⚠️  Ollamaサーバーが起動していません。"ollama serve" コマンドで起動してください。');
    }
});

// グレースフルシャットダウン
process.on('SIGTERM', async () => {
    console.log('📛 サーバーを終了しています...');
    await pool.end();
    process.exit(0);
});

process.on('SIGINT', async () => {
    console.log('📛 サーバーを終了しています...');
    await pool.end();
    process.exit(0);
});