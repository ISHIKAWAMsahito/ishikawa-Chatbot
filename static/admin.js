// ▼▼▼ Supabase初期化 ▼▼▼
    window.supabaseClient = null;

    async function initializeSupabase() {
        try {
            // supabaseライブラリがロードされているか確認
            if (typeof supabase === 'undefined') {
                console.error("Supabaseライブラリがロードされていません。");
                return;
            }

            const response = await fetch('/api/client/config');
            if (!response.ok) return;
            const config = await response.json();
            
            if (config.supabase_url && config.supabase_anon_key) {
                // const { createClient } = ... を使わず、直接 supabase.createClient を呼ぶ
                window.supabaseClient = supabase.createClient(config.supabase_url, config.supabase_anon_key);
                console.log('✅ Supabase初期化完了 (Admin)');
            }
        } catch (error) {
            console.error("Supabase初期化失敗:", error);
        }
    }
    initializeSupabase();

    document.addEventListener('DOMContentLoaded', () => {
        // ▼▼▼ ヘルパー関数 (XSS対策・安全な描画) ▼▼▼

        // 文字列からリンクを安全にDOM化する関数 (innerHTML不使用)
        function renderSafeText(container, text) {
            container.textContent = ''; // 内容をクリア
            
            // Markdown形式のリンク [text](url) を検出する正規表現
            const linkRe = /\[([^\]]*)\]\((https?:\/\/[^\)]+)\)/g;
            let lastIdx = 0;
            let match;

            while ((match = linkRe.exec(text)) !== null) {
                // マッチする前のテキストを追加
                if (match.index > lastIdx) {
                    container.appendChild(document.createTextNode(text.slice(lastIdx, match.index)));
                }
                
                // リンク要素を作成 (安全なプロパティ設定)
                const a = document.createElement('a');
                a.textContent = match[1];
                a.href = match[2]; 
                a.target = '_blank';
                a.rel = 'noopener';
                a.style.color = '#007bff';
                a.style.textDecoration = 'underline';
                container.appendChild(a);

                lastIdx = linkRe.lastIndex;
            }
            
            // 残りのテキストを追加
            if (lastIdx < text.length) {
                container.appendChild(document.createTextNode(text.slice(lastIdx)));
            }
        }

        function escapeHtml(s) {
            if (typeof s !== 'string') return '';
            const div = document.createElement('div');
            div.textContent = s;
            return div.innerHTML;
        }

        function isAllowedUrl(url) {
            try {
                const u = (url || '').trim();
                return u.startsWith('https://') || u.startsWith('http://');
            } catch (e) { return false; }
        }

        // ▼▼▼ DOM要素の取得 ▼▼▼
        const chatMessages = document.getElementById('chatMessages');
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const modelSelect = document.getElementById('modelSelect');
        const fileInput = document.getElementById('fileInput');
        const documentList = document.getElementById('documentList');
        const collectionSelect = document.getElementById('collectionSelect');
        const topKSelect = document.getElementById('topKSelect');
        const embeddingModelSelect = document.getElementById('embeddingModelSelect');
        const scrapeUrlInput = document.getElementById('scrapeUrlInput');
        const scrapeBtn = document.getElementById('scrapeBtn');
        const adminDarkBtn = document.getElementById('adminDarkModeBtn');

        let currentCollection = '';
        let backendConnected = false;
        let settingsSocket = null;
        let wsRetryCount = 0;
        let abortController = null; // 生成停止用

        // ▼▼▼ ダークモード設定 ▼▼▼
        if (localStorage.getItem('adminTheme') === 'dark') {
            document.body.classList.add('dark-mode');
            if(adminDarkBtn) adminDarkBtn.textContent = '☀️ モード切替';
        }

        if(adminDarkBtn) {
            adminDarkBtn.addEventListener('click', () => {
                document.body.classList.toggle('dark-mode');
                if (document.body.classList.contains('dark-mode')) {
                    localStorage.setItem('adminTheme', 'dark');
                    adminDarkBtn.textContent = '☀️ モード切替';
                } else {
                    localStorage.setItem('adminTheme', 'light');
                    adminDarkBtn.textContent = '🌙 モード切替';
                }
            });
        }

        // ▼▼▼ WebSocket接続 (設定同期) ▼▼▼
        function connectSettingsWebSocket() {
            try {
                fetch('/api/admin/system/ws-token', { credentials: 'include' })
                    .then(r => r.ok ? r.json() : Promise.reject(new Error('ws-token failed')))
                    .then(data => {
                        const token = data.token;
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${protocol}//${window.location.host}/ws/settings?token=${encodeURIComponent(token)}`;
                        console.log('📡 WebSocket接続試行中...');
                        settingsSocket = new WebSocket(wsUrl);
                        attachWebSocketHandlers();
                    })
                    .catch(err => {
                        console.warn('WebSocketトークン取得失敗:', err);
                        updateSyncStatus(false, '設定同期: 認証が必要です');
                        setTimeout(connectSettingsWebSocket, 5000);
                    });
            } catch (error) {
                console.error('WebSocket初期化エラー:', error);
                setTimeout(connectSettingsWebSocket, 5000);
            }
        }

        function attachWebSocketHandlers() {
            if (!settingsSocket) return;

            settingsSocket.onopen = function() {
                console.log('✅ 設定同期WebSocket接続完了');
                updateSyncStatus(true, '設定同期接続済み');
                wsRetryCount = 0;
            };

            settingsSocket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'settings_update') {
                        console.log('📥 設定更新受信:', data.data);
                        applyReceivedSettings(data.data);
                    }
                } catch (e) {
                    console.debug('WebSocketメッセージ:', event.data);
                }
            };

            settingsSocket.onclose = function() {
                console.log('⚠️ WebSocket切断');
                updateSyncStatus(false, '設定同期切断（再接続待機中）');
                wsRetryCount++;
                const delay = Math.min(1000 * Math.pow(2, wsRetryCount), 30000);
                setTimeout(() => {
                    if (!settingsSocket || settingsSocket.readyState === WebSocket.CLOSED) {
                        connectSettingsWebSocket();
                    }
                }, delay);
            };

            settingsSocket.onerror = function(error) {
                console.error('❌ WebSocketエラー');
            };
        }

        function updateSyncStatus(connected, message) {
            const syncStatus = document.getElementById('syncStatus');
            const syncStatusText = document.getElementById('syncStatusText');
            const autoSyncIndicator = document.getElementById('autoSyncIndicator');
            if (syncStatus && syncStatusText) {
                syncStatus.className = `status-indicator ${connected ? 'status-connected' : 'status-disconnected'}`;
                syncStatusText.textContent = message;
                if (autoSyncIndicator) {
                    autoSyncIndicator.style.display = connected ? 'inline-block' : 'none';
                }
            }
        }

        function applyReceivedSettings(settings) {
            if (settings.model && modelSelect.value !== settings.model) {
                if ([...modelSelect.options].some(opt => opt.value === settings.model)) {
                    modelSelect.value = settings.model;
                    highlightSettingChange('modelSetting', 'modelSyncStatus', 'モデルが更新されました');
                }
            }
            if (settings.collection && collectionSelect.value !== settings.collection) {
                if ([...collectionSelect.options].some(opt => opt.value === settings.collection)) {
                    collectionSelect.value = settings.collection;
                    currentCollection = settings.collection;
                    updateDocumentList();
                    highlightSettingChange('collectionSetting', 'collectionSyncStatus', 'コレクションが更新されました');
                }
            }
            if (settings.embedding_model && embeddingModelSelect.value !== settings.embedding_model) {
                if ([...embeddingModelSelect.options].some(opt => opt.value === settings.embedding_model)) {
                    embeddingModelSelect.value = settings.embedding_model;
                    highlightSettingChange('embeddingSetting', 'embeddingSyncStatus', '埋め込みモデルが更新されました');
                }
            }
            if (settings.top_k && parseInt(topKSelect.value) !== settings.top_k) {
                if ([...topKSelect.options].some(opt => parseInt(opt.value) === settings.top_k)) {
                    topKSelect.value = settings.top_k.toString();
                    highlightSettingChange('modelSetting', 'modelSyncStatus', '検索件数が更新されました');
                }
            }
        }

        function highlightSettingChange(settingId, statusId, message) {
            const settingElement = document.getElementById(settingId);
            const statusElement = document.getElementById(statusId);
            if (settingElement) {
                settingElement.classList.add('active');
                setTimeout(() => settingElement.classList.remove('active'), 2000);
            }
            if (statusElement) {
                statusElement.textContent = message;
                statusElement.className = 'sync-status';
                setTimeout(() => {
                    statusElement.textContent = '設定同期待機中...';
                    statusElement.className = 'sync-status';
                }, 3000);
            }
        }

        async function updateServerSettings() {
            const settings = {
                model: modelSelect.value,
                collection: collectionSelect.value,
                embedding_model: embeddingModelSelect.value,
                top_k: parseInt(topKSelect.value, 10)
            };
            try {
                await fetch('/api/admin/system/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });
            } catch (error) {
                console.error('設定送信通信エラー:', error);
            }
        }

        // ▼▼▼ チャットUI関連 ▼▼▼
        function addMessage(text, type) {
            const messageWrapper = document.createElement('div');
            messageWrapper.className = `message ${type}`;
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            // ★安全なテキスト描画 (innerHTML不使用)
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

        async function sendMessage() {
            const userInput = chatInput.value.trim();

            // 停止ロジック
            if (abortController) {
                abortController.abort();
                abortController = null;
                sendBtn.classList.remove('generating');
                sendBtn.innerHTML = '<span>送信</span><span>🔍</span>';
                
                const generatingMsgs = document.querySelectorAll('.message.bot');
                if(generatingMsgs.length > 0) {
                    const lastMsg = generatingMsgs[generatingMsgs.length - 1];
                    if(!lastMsg.dataset.finished) {
                        const contentDiv = lastMsg.querySelector('.message-content');
                        // DOM操作で追記
                        const stopSpan = document.createElement('span');
                        stopSpan.style.color = '#dc3545';
                        stopSpan.style.fontSize = '0.8rem';
                        stopSpan.style.fontWeight = 'bold';
                        stopSpan.textContent = '\n(ユーザーにより停止されました)';
                        contentDiv.appendChild(stopSpan);
                        lastMsg.dataset.finished = "true";
                    }
                }
                return;
            }

            if (!userInput) return;
            if (!currentCollection) {
                addMessage('❌ コレクションが選択されていません。', 'bot');
                return;
            }

            sendBtn.classList.add('generating');
            sendBtn.innerHTML = '<span>停止</span><span>⏹</span>';

            abortController = new AbortController();
            const signal = abortController.signal;

            addMessage(userInput, 'user');
            chatInput.value = '';
            chatInput.style.height = 'auto';

            const botMessageElement = addMessage('思考中...', 'bot');
            const botMessageContent = botMessageElement.querySelector('.message-content');
            let fullResponse = '';

            try {
                const response = await fetch(`/api/admin/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: userInput,
                        model: modelSelect.value,
                        collection: currentCollection,
                        embedding_model: embeddingModelSelect.value,
                        top_k: parseInt(topKSelect.value, 10)
                    }),
                    signal: signal
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: `APIエラー: ${response.statusText}` }));
                    throw new Error(errorData.detail);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                let isFirstChunk = true; // 初回フラグ

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        botMessageElement.dataset.finished = "true";
                        // フィードバックボタンの追加 (DOM操作)
                        if (botMessageElement.dataset.logId) {
                            const feedbackContainer = document.createElement('div');
                            feedbackContainer.className = 'feedback-container';
                            
                            const span = document.createElement('span');
                            span.textContent = 'この回答は役に立ちましたか？';
                            feedbackContainer.appendChild(span);

                            const btnGood = document.createElement('button');
                            btnGood.className = 'feedback-btn';
                            btnGood.dataset.rating = 'good';
                            btnGood.textContent = '👍';
                            feedbackContainer.appendChild(btnGood);

                            const btnBad = document.createElement('button');
                            btnBad.className = 'feedback-btn';
                            btnBad.dataset.rating = 'bad';
                            btnBad.textContent = '👎';
                            feedbackContainer.appendChild(btnBad);
                            
                            botMessageElement.appendChild(feedbackContainer);
                        }
                        break;
                    }
                    
                    let buffer = decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    
                    for (const line of lines) {
                        if (line.trim().startsWith('data: ')) {
                            const jsonString = line.substring(6).trim();
                            if (jsonString) {
                                try {
                                    const data = JSON.parse(jsonString);
                                    if (data.feedback_id) {
                                        botMessageElement.dataset.feedbackId = data.feedback_id;
                                        botMessageElement.dataset.logId = data.feedback_id;
                                    }

                                    // ステータスメッセージ (DOM操作)
                                    if (data.status_message) {
                                        botMessageContent.textContent = ''; // クリア
                                        const span = document.createElement('span');
                                        span.className = 'status-pulse';
                                        span.textContent = data.status_message;
                                        botMessageContent.appendChild(span);
                                        if(chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
                                        continue; 
                                    }

                                    // 回答本文 (DOM操作 + renderSafeText)
                                    if (data.content) {
                                        if (isFirstChunk) {
                                            botMessageContent.textContent = ''; // クリア
                                            isFirstChunk = false;
                                        }

                                        fullResponse += data.content;
                                        // ★安全な描画関数を使用 (Snyk XSS対策)
                                        renderSafeText(botMessageContent, fullResponse);
                                        
                                        if(chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
                                    }

                                } catch (e) {
                                    console.error("Stream parse error:", e);
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                if (error.name === 'AbortError') {
                    console.log('Fetch aborted by admin');
                } else {
                    console.error('メッセージ送信エラー:', error);
                    botMessageContent.textContent = ''; // クリア
                    const errorSpan = document.createElement('span');
                    errorSpan.style.color = '#dc3545';
                    errorSpan.style.fontWeight = 'bold';
                    
                    if (error.message.includes("503") || error.message.includes("overloaded")) {
                        errorSpan.textContent = '⚠️ Gemini APIが過負荷状態です。しばらく待ってから再試行してください。';
                    } else {
                        errorSpan.textContent = `エラーが発生しました: ${error.message}`;
                    }
                    botMessageContent.appendChild(errorSpan);
                }
            } finally {
                if (abortController) {
                    abortController = null;
                    sendBtn.classList.remove('generating');
                    sendBtn.innerHTML = '<span>送信</span><span>🔍</span>';
                }
            }
        }

        // ▼▼▼ その他機能 ▼▼▼
        async function sendFeedback(element, rating) {
            try {
                if (!window.supabaseClient) return;
                const { error } = await window.supabaseClient
                    .from('anonymous_comments')
                    .insert([{
                        rating: rating,
                        comment: "管理者画面からの評価",
                        created_at: new Date().toISOString()
                    }]);

                if (error) throw error;
                const feedbackContainer = element.parentElement;
                feedbackContainer.innerHTML = '<span class="feedback-thanks">フィードバックを保存しました</span>';
            } catch(error) {
                console.error("フィードバック送信エラー:", error);
                alert("送信に失敗しました。");
            }
        }

        async function handleFileUpload(event) {
            const files = Array.from(event.target.files);
            if (files.length === 0 || !currentCollection) {
                addMessage(files.length === 0 ? 'ファイルが選択されていません。' : '❌ コレクションを選択してください。', 'bot');
                return;
            }
            for (const file of files) {
                const botMsg = addMessage(`⏳「${file.name}」を処理中...`, 'bot');
                const formData = new FormData();
                formData.append('file', file);
                formData.append('collection_name', currentCollection);
                formData.append('embedding_model', embeddingModelSelect.value);
                try {
                    const response = await fetch(`/api/admin/documents/upload`, { method: 'POST', body: formData });
                    const result = await response.json();
                    if (response.ok) {
                        botMsg.querySelector('.message-content').textContent = `✅「${file.name}」をコレクション「${currentCollection}」に追加しました（${result.chunks}チャンク）`;
                        await refreshCollections();
                    } else {
                        botMsg.querySelector('.message-content').textContent = `❌「${file.name}」の処理中にエラー: ${result.detail}`;
                    }
                } catch (error) {
                    botMsg.querySelector('.message-content').textContent = `❌「${file.name}」の処理中に通信エラー: ${error.message}`;
                }
            }
            fileInput.value = '';
        }

        async function createCollection() {
            const collectionName = document.getElementById('newCollectionName').value.trim();
            if (!collectionName) {
                addMessage('❌ コレクション名を入力してください。', 'bot');
                return;
            }
            try {
                const response = await fetch(`/api/admin/system/collections`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: collectionName })
                });
                const result = await response.json();
                addMessage(response.ok ? `✅ コレクション「${collectionName}」を作成しました` : `❌ 作成失敗: ${result.detail}`, 'bot');
                if (response.ok) {
                    document.getElementById('newCollectionName').value = '';
                    setTimeout(refreshCollections, 500);
                }
            } catch (error) {
                addMessage(`❌ 作成中に通信エラー: ${error.message}`, 'bot');
            }
        }

        async function deleteCollection() {
            const collectionName = collectionSelect.value;
            if (!collectionName) return;
            const confirmed = await showConfirm(`コレクション「${collectionName}」を本当に削除しますか？`);
            if (!confirmed) return;
            try {
                const response = await fetch(`/api/admin/system/collections/${collectionName}`, { method: 'DELETE' });
                const result = await response.json();
                addMessage(response.ok ? `✅ ${result.message}` : `❌ 削除失敗: ${result.detail}`, 'bot');
                if (response.ok) await refreshCollections();
            } catch (error) {
                addMessage(`❌ 削除中に通信エラー: ${error.message}`, 'bot');
            }
        }

        async function refreshCollections() {
            try {
                const response = await fetch(`/api/admin/system/collections`);
                if (response.ok) {
                    const collections = await response.json();
                    updateCollectionList(collections);
                } else {
                    addMessage('❌ コレクション一覧の取得に失敗しました。', 'bot');
                    updateCollectionList([]);
                }
            } catch (error) {
                addMessage(`❌ コレクションの更新に失敗: ${error.message}`, 'bot');
                updateCollectionList([]);
            }
        }

        async function updateDocumentList() {
            if (!currentCollection) {
                // 【修正】innerHTMLを使わずDOM操作でメッセージを設定
                documentList.textContent = '';
                const span = document.createElement('span');
                span.textContent = 'コレクションを選択してください';
                documentList.appendChild(span);
                return;
            }
            
            documentList.textContent = '読み込み中...'; // 【修正】textContentを使用
            
            try {
                const response = await fetch(`/api/admin/documents/collections/${currentCollection}/documents`);
                const data = await response.json();
                
                documentList.textContent = ''; // 【修正】textContentでクリア
                
                if (response.ok && data.documents && data.documents.length > 0) {
                    data.documents.forEach(doc => {
                        const item = document.createElement('div');
                        item.className = 'document-item';
                        item.textContent = doc.id;
                        documentList.appendChild(item);
                    });

                    // ▼▼▼ Snyk指摘箇所の修正 (ここがメインの修正) ▼▼▼
                    /* 修正前:
                       totalItem.innerHTML = `<strong>合計: ${data.count} チャンク</strong>`;
                    */
                    const totalItem = document.createElement('div');
                    totalItem.className = 'total-chunks-info'; 
                    
                    const strong = document.createElement('strong');
                    // textContentならスクリプトが含まれていても実行されず安全
                    strong.textContent = `合計: ${data.count} チャンク`; 
                    
                    totalItem.appendChild(strong);
                    documentList.appendChild(totalItem);
                    // ▲▲▲ 修正ここまで ▲▲▲

                } else {
                    // 【修正】innerHTMLを使わずDOM操作
                    const span = document.createElement('span');
                    span.textContent = 'ドキュメントはありません';
                    documentList.appendChild(span);
                }
            } catch (error) {
                // 【修正】innerHTMLを使わずDOM操作
                documentList.textContent = '';
                const span = document.createElement('span');
                span.textContent = 'リストの取得中にエラーが発生しました';
                documentList.appendChild(span);
            }
        }

        function updateGenericList(items, selectElement, defaultValue = '') {
            const currentValue = selectElement.value;
            selectElement.innerHTML = '';
            items.forEach(item => {
                const option = document.createElement('option');
                option.value = typeof item === 'string' ? item : item.name;
                option.textContent = typeof item === 'string' ? item : `${item.name} (${item.count})`;
                selectElement.appendChild(option);
            });
            const values = items.map(item => typeof item === 'string' ? item : item.name);
            if (values.includes(currentValue)) selectElement.value = currentValue;
            else if (values.includes(defaultValue)) selectElement.value = defaultValue;
            else if (values.length > 0) selectElement.value = values[0];
            return selectElement.value || '';
        }

        function updateCollectionList(collections) {
            currentCollection = updateGenericList(collections, collectionSelect);
            updateDocumentList();
        }

        async function scrapeWebsite() {
            const url = scrapeUrlInput.value.trim();
            if (!url || !currentCollection) {
                addMessage(!url ? '❌ URLを入力してください。' : '❌ コレクションを選択してください。', 'bot');
                return;
            }
            scrapeBtn.disabled = true;
            scrapeBtn.textContent = '取得中...';
            addMessage(`⏳ URLから情報を取得中...\n${url}`, 'bot');
            try {
                const response = await fetch(`/api/admin/documents/scrape`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        url: url,
                        collection_name: currentCollection,
                        embedding_model: embeddingModelSelect.value
                    })
                });
                const result = await response.json();
                addMessage(response.ok ? `✅ URLの情報を追加しました（${result.chunks}チャンク）` : `❌ 取得失敗: ${result.detail}`, 'bot');
                if (response.ok) await refreshCollections();
            } catch (error) {
                addMessage(`❌ 通信エラー: ${error.message}`, 'bot');
            } finally {
                scrapeBtn.disabled = false;
                scrapeBtn.textContent = 'このURLから情報を取得';
                scrapeUrlInput.value = '';
            }
        }

        function showConfirm(message) {
            return new Promise(resolve => {
                const modal = document.getElementById('confirmModal');
                const msgEl = document.getElementById('modalMessage');
                const confirmBtn = document.getElementById('modalConfirm');
                const cancelBtn = document.getElementById('modalCancel');

                msgEl.textContent = message;
                modal.classList.add('visible');
                const close = (value) => {
                    modal.classList.remove('visible');
                    confirmBtn.onclick = null;
                    cancelBtn.onclick = null;
                    resolve(value);
                };
                confirmBtn.onclick = () => close(true);
                cancelBtn.onclick = () => close(false);
            });
        }

        // ▼▼▼ 接続チェックと初期化 ▼▼▼
        async function checkBackendConnection() {
            try {
                const response = await fetch(`/health`);
                const data = await response.json();
                backendConnected = response.ok;
                if (!backendConnected) addMessage('❌ バックエンドサーバーに接続できません。', 'bot');
                return data.database; 
            } catch (error) {
                backendConnected = false;
                addMessage('❌ バックエンドサーバーに接続できません。', 'bot');
                return null;
            }
        }

        function updateStatus(elementId, textId, connected, serviceName) {
            const statusIndicator = document.getElementById(elementId);
            const statusText = document.getElementById(textId);
            if(statusIndicator && statusText) {
                statusIndicator.className = `status-indicator ${connected ? 'status-connected' : 'status-disconnected'}`;
                statusText.textContent = connected ? `${serviceName} 接続済み` : `${serviceName} 未接続`;
            }
        }

        async function checkGeminiConnection() {
            try {
                const response = await fetch(`/api/admin/system/gemini/status`);
                const data = await response.json();
                updateStatus('geminiStatus', 'geminiStatusText', response.ok && data.connected, 'Gemini API');
                if (response.ok && data.connected && data.models) {
                    updateGenericList(data.models.map(m => m), modelSelect, 'gemini-2.5-flash');
                }
            } catch (error) {
                updateStatus('geminiStatus', 'geminiStatusText', false, 'Gemini API');
            }
        }

        async function initializeSystem() {
            const backendUrlEl = document.getElementById('backendUrl');
            if (backendUrlEl) backendUrlEl.textContent = window.location.origin;
            
            const dbType = await checkBackendConnection();
            if (backendConnected) {
                const dbName = dbType || '不明';
                updateStatus('dbStatusIndicator', 'dbStatusText', true, dbName.charAt(0).toUpperCase() + dbName.slice(1));
                const collectionTitle = document.getElementById('collectionTitle');
                if(collectionTitle) collectionTitle.textContent = `🗃️ ${dbName.charAt(0).toUpperCase() + dbName.slice(1)} コレクション`;
                
                await checkGeminiConnection();
                await refreshCollections();
                connectSettingsWebSocket();
            }
        }

        // ▼▼▼ グローバル関数登録 (HTMLから呼び出し用) ▼▼▼
        window.createCollection = createCollection;
        window.deleteCollection = deleteCollection;
        window.refreshCollections = refreshCollections;
        window.scrapeWebsite = scrapeWebsite;
        window.openDocumentManager = function() { window.open('DB.html', '_blank'); };
        window.setScrapeUrl = function(url) { scrapeUrlInput.value = url; }

        // ▼▼▼ イベントリスナー登録 ▼▼▼
        modelSelect.addEventListener('change', updateServerSettings);
        collectionSelect.addEventListener('change', (e) => {
            currentCollection = e.target.value;
            updateDocumentList();
            updateServerSettings();
        });
        embeddingModelSelect.addEventListener('change', updateServerSettings);
        topKSelect.addEventListener('change', updateServerSettings);
        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        fileInput.addEventListener('change', handleFileUpload);
        chatMessages.addEventListener('click', (e) => {
            if (e.target.classList.contains('feedback-btn')) {
                const rating = e.target.dataset.rating;
                if (rating) sendFeedback(e.target, rating);
            }
        });

        // ▼▼▼ 画像表示モーダル機能 (XSS対策済み) ▼▼▼
        window.showSourceImage = async function(sourceName) {
            if (!window.supabaseClient) {
                alert('Supabaseクライアントが初期化されていません');
                return;
            }

            const modal = document.getElementById('imageModal');
            const content = document.getElementById('imageModalContent');
            
            modal.classList.add('visible');
            content.textContent = '画像を読み込み中...'; // innerHTMLを使わずクリア

            try {
                const baseName = sourceName.replace(/\.(pdf|docx|txt)$/i, '');
                const imagePaths = [
                    `converted_images_rules/${baseName}_001.jpg`,
                    `converted_images_common/${baseName}_001.jpg`,
                    `${baseName}_001.jpg`,
                    `${baseName}.jpg`
                ];

                let imageUrl = null;
                for (const path of imagePaths) {
                    const { data, error } = await window.supabaseClient.storage.from('images').getPublicUrl(path);
                    if (!error && data) {
                        const imageLoaded = await new Promise((resolve) => {
                            const img = new Image();
                            img.onload = () => resolve(true);
                            img.onerror = () => resolve(false);
                            img.src = data.publicUrl;
                        });
                        if (imageLoaded) {
                            imageUrl = data.publicUrl;
                            displayImage(imageUrl);
                            return; 
                        }
                    }
                }

                // 見つからない場合、検索
                const { data: files } = await window.supabaseClient.storage.from('images').list('', { search: baseName });
                if (files && files.length > 0) {
                    const { data: urlData } = await window.supabaseClient.storage.from('images').getPublicUrl(files[0].name);
                    displayImage(urlData.publicUrl);
                } else {
                    // ★安全なエラー表示 (DOM操作)
                    content.textContent = '';
                    const errDiv = document.createElement('div');
                    errDiv.className = 'image-modal-error';
                    errDiv.textContent = '画像が見つかりませんでした';
                    const br = document.createElement('br');
                    const small = document.createElement('small');
                    small.textContent = 'Source: ' + sourceName;
                    errDiv.appendChild(br);
                    errDiv.appendChild(small);
                    content.appendChild(errDiv);
                }
            } catch (error) {
                console.error('画像取得エラー:', error);
                content.textContent = '';
                const errDiv = document.createElement('div');
                errDiv.className = 'image-modal-error';
                errDiv.textContent = 'エラーが発生しました';
                content.appendChild(errDiv);
            }
        };

        function displayImage(imageUrl) {
            const content = document.getElementById('imageModalContent');
            content.textContent = '';

            if (!imageUrl || !isAllowedUrl(imageUrl)) {
                const errDiv = document.createElement('div');
                errDiv.className = 'image-modal-error';
                errDiv.textContent = '無効な画像URLです。';
                content.appendChild(errDiv);
                return;
            }
            
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = "参照元画像";
            img.className = "image-modal-image";
            
            img.onerror = function() {
                const parent = this.parentElement;
                parent.textContent = '';
                const errDiv = document.createElement('div');
                errDiv.className = 'image-modal-error';
                errDiv.textContent = '画像を読み込めませんでした';
                parent.appendChild(errDiv);
            };

            content.appendChild(img);
        }

        window.closeImageModal = function() {
            document.getElementById('imageModal').classList.remove('visible');
        };

        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') window.closeImageModal(); });
        document.getElementById('imageModal').addEventListener('click', function(e) { if (e.target === this) window.closeImageModal(); });

        // 初期化実行
        initializeSystem();
    });