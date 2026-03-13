document.addEventListener('DOMContentLoaded', 
    function() {
        // --- 変数定義 ---
        const BACKEND_API_BASE_URL = window.location.origin;
        const chatMessages = document.getElementById('chatMessages');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const openCommentModalBtn = document.getElementById('openCommentModalBtn');
        const commentModal = document.getElementById('commentModal');
        const closeBtn = document.querySelector('.modal-close-btn');
        const sendCommentBtn = document.getElementById('sendCommentBtn');
        const commentTextarea = document.getElementById('commentTextarea');
        const successMessage = document.getElementById('successMessage');
        const errorMessage = document.getElementById('errorMessage');
        const themeButtons = document.querySelectorAll('.theme-btn');
        const bodyElement = document.body;

        let settingsSocket = null;
        let abortController = null; // 生成停止用のコントローラー
function renderSafeText(container, text) {
            container.textContent = ''; // 内容をクリア
            
            // ★修正: ファイル名の中に "]" が含まれていてもリンクとして認識できるよう柔軟にしました
            const linkRe = /\[(.*?)\]\((https?:\/\/[^\)]+)\)/g;
            let lastIdx = 0;
            let match;

            while ((match = linkRe.exec(text)) !== null) {
                if (match.index > lastIdx) {
                    container.appendChild(document.createTextNode(text.slice(lastIdx, match.index)));
                }
                
                const a = document.createElement('a');
                a.textContent = match[1];
                if (isAllowedUrl(match[2])) {
                    a.href = match[2]; 
                    a.target = '_blank';
                    a.rel = 'noopener';
                    a.style.color = '#007bff';
                    a.style.textDecoration = 'underline';
                } else {
                    a.href = "#"; // 無効なURLの保護
                }
                container.appendChild(a);

                lastIdx = linkRe.lastIndex;
            }
            
            if (lastIdx < text.length) {
                container.appendChild(document.createTextNode(text.slice(lastIdx)));
            }
        }
        // ダークモード切替機能
const darkModeBtn = document.getElementById('darkModeBtn');
const body = document.body;

// 保存された設定を読み込む
if (localStorage.getItem('theme') === 'dark') {
    body.classList.add('dark-mode');
    darkModeBtn.textContent = '☀️'; // 太陽アイコンに変更
}

darkModeBtn.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    if (body.classList.contains('dark-mode')) {
        localStorage.setItem('theme', 'dark');
        darkModeBtn.textContent = '☀️';
    } else {
        localStorage.setItem('theme', 'light');
        darkModeBtn.textContent = '🌙';
    }
});

        // --- テーマ変更機能 ---
        const themes = {
            default: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            green: 'linear-gradient(135deg, #2AF598 0%, #009EFD 100%)',
            orange: 'linear-gradient(135deg, #F6D365 0%, #FDA085 100%)',
            pink: 'linear-gradient(135deg, #F093FB 0%, #F5576C 100%)'
        };

        if (themeButtons) {
            themeButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const themeName = button.dataset.theme;
                    if (themes[themeName]) {
                        bodyElement.style.background = themes[themeName];
                    }
                });
            });
        }

        // --- コメントモーダル機能 ---
        if (openCommentModalBtn && commentModal) {
            openCommentModalBtn.onclick = () => {
                commentModal.style.display = 'block';
                commentTextarea.value = '';
                successMessage.style.display = 'none';
                errorMessage.style.display = 'none';
            }
        }

        if (closeBtn && commentModal) {
            closeBtn.onclick = () => { commentModal.style.display = 'none'; }
        }

        window.onclick = (event) => {
            if (commentModal && event.target == commentModal) {
                commentModal.style.display = 'none';
            }
        }
        if (sendCommentBtn) {
            sendCommentBtn.onclick = async () => {
                const comment = commentTextarea.value.trim();
                if (comment === '') {
                    alert('コメントを入力してください。');
                    return;
                }

                sendCommentBtn.disabled = true;
                successMessage.style.display = 'none';
                errorMessage.style.display = 'none';

                try {
                    // バックエンドAPI経由で送信し、サーバ側で匿名コメントを保存＆自動ベクトル化
                    const payload = {
                        feedback_id: "anonymous",      // チャットログに紐づかない純粋なコメント
                        rating: "comment_only",        // 種別を判別しやすい任意のラベル
                        comment: comment
                    };

                    const response = await fetch(`${BACKEND_API_BASE_URL}/api`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                        credentials: 'include', // セッション認証用
                    });

                    if (!response.ok) {
                        const data = await response.json().catch(() => ({}));
                        const msg = data.detail || `コメント送信に失敗しました (status: ${response.status})`;
                        throw new Error(msg);
                    }

                    successMessage.style.display = 'block';
                    commentTextarea.value = '';

                    setTimeout(() => {
                        commentModal.style.display = 'none';
                        successMessage.style.display = 'none';
                    }, 2000);

                } catch (error) {
                    console.error('❌ コメント送信エラー:', error);
                    errorMessage.style.display = 'block';
                } finally {
                    sendCommentBtn.disabled = false;
                }
            };
        }

        // --- WebSocket機能 ---
        function connectSettingsWebSocket() {
            try {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                
                // 学生用のエンドポイントに接続
                const wsUrl = `${protocol}//${window.location.host}/ws/client/settings`;
                
                settingsSocket = new WebSocket(wsUrl);
                
                settingsSocket.onopen = () => {
                    console.log('設定同期WebSocket接続完了');
                };
                
                settingsSocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'settings_update') {
                            console.log('設定更新を受信:', data.data);
                            // 緑の通知バナーだけ表示する
                            showSettingsUpdateNotification();
                        }
                    } catch (e) {
                        console.error('WebSocketメッセージ解析エラー:', e);
                    }
                };
                
                settingsSocket.onclose = () => {
                    console.log('設定同期WebSocket切断');
                    // ★削除: updateSyncIndicator(false, '設定同期切断');
                    // 自動再接続のみ行う
                    setTimeout(connectSettingsWebSocket, 5000);
                };
                
                settingsSocket.onerror = (error) => {
                    console.error('設定同期WebSocketエラー:', error);
                    // ★削除: updateSyncIndicator(false, '設定同期エラー');
                    settingsSocket.close();
                };
                
            } catch (error) {
                console.error('WebSocket接続エラー:', error);
                setTimeout(connectSettingsWebSocket, 5000);
            }
        }

        function updateSyncIndicator(connected, message) {
            const syncDot = document.getElementById('syncDot');
            const syncText = document.getElementById('syncText');
            if (syncDot && syncText) {
                syncDot.className = connected ? 'sync-dot' : 'sync-dot disconnected';
                syncText.textContent = message;
            }
        }

        function showSettingsUpdateNotification() {
            const notification = document.getElementById('settingsNotification');
            if (notification) {
                notification.classList.add('show');
                setTimeout(() => hideSettingsNotification(), 5000);
            }
        }

        function hideSettingsNotification() {
            const notification = document.getElementById('settingsNotification');
            if (notification) {
                notification.classList.remove('show');
            }
        }
        window.hideSettingsNotification = hideSettingsNotification;

        function addMessage(text, type) {
            if (!chatMessages) return null;

            const messageWrapper = document.createElement('div');
            messageWrapper.className = `message ${type}`;
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            // ★修正: 不要な変数 out を削除し、安全な描画関数のみを使用します
            renderSafeText(messageContent, text);
            
            messageWrapper.appendChild(messageContent);
            const messageTime = document.createElement('div');
            messageTime.className = 'message-time';
            messageTime.textContent = new Date().toLocaleTimeString();
            messageWrapper.appendChild(messageTime);
            chatMessages.appendChild(messageWrapper);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return messageWrapper;
        }

        // --- フィードバック送信機能 ---
        async function sendFeedback(element, rating) {
            console.log(`フィードバック送信開始: ${rating}`);
            try {
                if (!window.supabaseClient) {
                    console.error("Supabaseクライアントが初期化されていません");
                    alert("通信エラー: フィードバックを送信できませんでした。");
                    return;
                }

                const insertData = {
                    rating: rating,
                    comment: "ボタンによる評価",
                    created_at: new Date().toISOString()
                };

                const { data, error } = await window.supabaseClient
                    .from('anonymous_comments')
                    .insert([insertData]);

                if (error) {
                    throw error;
                }

                console.log("Supabaseへの保存成功:", data);
                const feedbackContainer = element.parentElement;
                feedbackContainer.innerHTML = '<span style="font-size: 0.8rem; color: #28a745;">フィードバックありがとうございます!</span>';
            } catch(error) {
                console.error("フィードバック送信エラー:", error);
                alert("送信に失敗しました。");
            }
        }

        // --- チャットメッセージ送信機能 (停止機能付き) ---
        async function sendMessage() {
            const userInput = chatInput.value.trim();

            // ============================================================
            // 【停止ロジック】すでに生成中(コントローラーがある)場合
            // ============================================================
            if (abortController) {
                abortController.abort(); // 通信を強制切断
                abortController = null;  // リセット
                // UIを「送信」ボタンに戻す
                sendBtn.classList.remove('generating');
                sendBtn.innerHTML = '<span>送信</span><span>📤</span>';
                // メッセージに「(停止しました)」と追記
                const generatingMsgs = document.querySelectorAll('.message.bot');
                if(generatingMsgs.length > 0) {
                    const lastMsg = generatingMsgs[generatingMsgs.length - 1];
                    // まだ完了フラグがない場合のみ追記
                    if(!lastMsg.dataset.finished) {
                        const contentDiv = lastMsg.querySelector('.message-content');
                        contentDiv.innerHTML += '<br><span style="color:#dc3545; font-size:0.8rem; font-weight:bold;">(ユーザーにより停止されました)</span>';
                        lastMsg.dataset.finished = "true";
                    }
                }
                return; // ここで処理を終了
            }

            // ============================================================
            // 【送信ロジック】ここからは通常の送信処理
            // ============================================================
            if (!userInput || !chatInput) return;

            // UI: ボタンを「停止」モードに変更(赤色にするクラスを追加)
            sendBtn.classList.add('generating');
            sendBtn.innerHTML = '<span>停止</span><span>⏹</span>'; // ■のアイコンに変更

            // AbortControllerの初期化
            abortController = new AbortController();
            const signal = abortController.signal; // これをfetchに渡す

            // ユーザーのメッセージを表示
            addMessage(userInput, 'user');
            chatInput.value = '';
            chatInput.style.height = 'auto';

            // ボットの応答準備
            const botMessageElement = addMessage('考え中...', 'bot');
            // addMessageが失敗した場合のガード
            if (!botMessageElement) {
                // ボタンを戻して終了
                sendBtn.classList.remove('generating');
                sendBtn.innerHTML = '<span>送信</span><span>📤</span>';
                abortController = null;
                return;
            }

            const botMessageContent = botMessageElement.querySelector('.message-content');
            let fullResponse = '';

            try {
                // APIリクエスト
                const response = await fetch(`${BACKEND_API_BASE_URL}/api/client/chat/stream`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: userInput
                    }),
                    signal: signal // ★重要: これにより中断が可能になります
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: `APIエラー: ${response.statusText}` }));
                    throw new Error(errorData.detail);
                }

                // レスポンスのストリーミング読み込み
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                botMessageContent.innerHTML = '';
                let buffer = '';
let isFirstChunk = true; // ★追加: これを while ループの前に置くのが重要
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        // 完了処理
                        botMessageElement.dataset.finished = "true";

                        // 回答完了時にフィードバックボタンを表示
                        if (botMessageElement.dataset.logId) {
                            const feedbackContainer = document.createElement('div');
                            feedbackContainer.className = 'feedback-container';
                            feedbackContainer.innerHTML = `
                                <span>この回答は役に立ちましたか?</span>
                                <button class="feedback-btn" data-rating="good">👍</button>
                                <button class="feedback-btn" data-rating="bad">👎</button>
                            `;
                            botMessageElement.appendChild(feedbackContainer);
                        }
                        break;
                    }

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.trim().startsWith('data: ')) {
                            const jsonString = line.substring(6).trim();
                            if (jsonString) {
                                try {
                            const data = JSON.parse(jsonString);
                            // ID保存 (既存)
                            if (data.feedback_id) {
                                botMessageElement.dataset.feedbackId = data.feedback_id;
                                botMessageElement.dataset.logId = data.feedback_id;
                            }
                            // --- ★ここから追加: ステータスメッセージの処理 ---
                            if (data.status_message) {
                                // ステータスメッセージで中身を上書き (XSS対策: エスケープ)
                                botMessageContent.innerHTML = '<span class="status-pulse">' + escapeHtml(data.status_message) + '</span>';
                                // チャット画面を最下部へスクロール
                                if(chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
                                continue; // コンテンツ処理はスキップして次のデータを待つ
                            }
                            // --- ここまで ---

                            // --- ★ここから変更: コンテンツ(回答本文)の処理 ---
                            if (data.content) {
                                if (isFirstChunk) {
                                    botMessageContent.textContent = '';
                                    isFirstChunk = false;
                                }

                                fullResponse += data.content;
                                // 関数を使って安全に描画
                                renderSafeText(botMessageContent, fullResponse);
                                
                                if(chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                            // --- ここまで ---
                        } catch (e) {
                            console.error("Error parsing stream data:", e);
                        }
                            }
                        }
                    }
                }
            } catch (error) {
                // エラーハンドリング
                if (error.name === 'AbortError') {
                    console.log('Fetch aborted by user');
                    // UI更新は冒頭のifブロックで行われるため、ここでは何もしない
                } else {
                    console.error('メッセージ送信エラー:', error);
                    botMessageContent.textContent = `エラーが発生しました: ${error.message}`;
                }
            } finally {
                // ============================================================
                // 【終了処理】正常完了した場合のボタンリセット
                // ============================================================
                // まだコントローラーが存在する!(中断ではなく)最後まで読み切った場合
                if (abortController) {
                    abortController = null;
                    sendBtn.classList.remove('generating');
                    sendBtn.innerHTML = '<span>送信</span><span>📤</span>';
                }
            }
        }

        // --- イベントリスナー設定 (Nullチェック付き) ---
        // フィードバックボタンのイベント委譲
        if (chatMessages) {
            chatMessages.addEventListener('click', function(e) {
                if (e.target.classList.contains('feedback-btn')) {
                    const rating = e.target.dataset.rating;
                    sendFeedback(e.target, rating);
                }
            });
        } else {
            console.error("❌ Error: chatMessages element not found in DOM.");
        }

        // 送信ボタンクリックイベント
        if (sendBtn) {
            sendBtn.addEventListener('click', sendMessage);
        } else {
            console.error("❌ Error: sendBtn element not found in DOM.");
        }

        // 入力エリアのイベント
        if (chatInput) {
            chatInput.addEventListener('input', () => {
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
            });

            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        } else {
            console.error("❌ Error: chatInput element not found in DOM.");
        }

        // --- 画像表示機能(署名付きURL対応) ---
        window.showSourceImage = function(signedUrl) {
            // 署名付きURLが直接渡される場合(chat_logic.pyから生成)
            const modal = document.getElementById('imageModal');
            const content = document.getElementById('imageModalContent');
            
            modal.classList.add('visible');
            content.innerHTML = '<div class="image-modal-loading">画像を読み込み中...</div>';

            // 署名付きURLを直接使用して画像を表示
            displayImage(signedUrl);
        };

        function displayImage(imageUrl) {
            const content = document.getElementById('imageModalContent');
            // XSS対策: 画像URLは https/http のみ許可
            if (!imageUrl || !isAllowedUrl(imageUrl)) {
                content.innerHTML = '<div class="image-modal-error">無効な画像URLです。</div>';
                return;
            }
            const safeUrl = escapeHtml(imageUrl);
            content.innerHTML = '<img src="' + safeUrl + '" alt="参照元画像" class="image-modal-image" onerror="this.parentElement.innerHTML=\'<div class=&#39;image-modal-error&#39;>画像の読み込みに失敗しました。</div>\'">';
        }

        window.closeImageModal = function() {
            const modal = document.getElementById('imageModal');
            modal.classList.remove('visible');
        };

        // ESCキーでモーダルを閉じる
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeImageModal();
            }
        });

        // モーダル背景クリックで閉じる
        document.getElementById('imageModal').addEventListener('click', function(e) {
            if (e.target === this) {
                closeImageModal();
            }
        });

    }); // End of DOMContentLoaded

    console.log("✅ スクリプトが読み込まれました");