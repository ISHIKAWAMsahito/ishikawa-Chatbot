// ==================================================
// 設定
// ==================================================
marked.setOptions({ breaks: true, gfm: true });
let chartInstance = null;
let abortController = null;

// ==================================================
// Toast 通知
// ==================================================
function showToast(msg, type = 'info', duration = 3500) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = `toast ${type} show`;
    setTimeout(() => { t.classList.remove('show'); }, duration);
}

// ==================================================
// 1. フィードバックデータ取得
// ==================================================
// ==================================================
// 1. フィードバックデータ取得（エラー時のcolspan修正）
// ==================================================
async function fetchData() {
    try {
        const res = await fetch('/api/admin/stats/data');
        if (!res.ok) {
            if (res.status === 401 || res.status === 403) { location.href = '/login'; return; }
            throw new Error(res.statusText);
        }
        const data = await res.json();
        renderDashboard(data);
    } catch (e) {
        showToast('データ取得失敗: ' + e.message, 'error');
        
        const tbody = document.getElementById('tableBody');
        tbody.innerHTML = '';

        const tr = document.createElement('tr');
        const td = document.createElement('td');
        
        td.colSpan = 4; // ★ 3から4に変更
        td.style.textAlign = 'center';
        td.style.color = '#dc3545';
        td.style.padding = '20px';
        td.textContent = 'エラー: ' + e.message; 
        
        tr.appendChild(td);
        tbody.appendChild(tr);
    }
}

function renderDashboard(data) {
    const total  = data.length;
    const good   = data.filter(i => i.rating === 'good').length;
    const bad    = data.filter(i => i.rating === 'bad').length;
    const rate   = total > 0 ? Math.round(good / total * 100) : 0;
    const week   = new Date(); week.setDate(week.getDate() - 7);
    const weekly = data.filter(i => new Date(i.created_at) > week).length;

    document.getElementById('kpiTotal').textContent    = total + '件';
    document.getElementById('kpiGoodRate').textContent = rate + '%';
    document.getElementById('kpiGoodSub').textContent  = `Good: ${good} / Bad: ${bad}`;
    document.getElementById('kpiWeekly').textContent   = weekly + '件';

    // テーブル描画
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    if (data.length === 0) {
        // ★ colspanを4に変更
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:20px;color:#aaa">データがありません</td></tr>';
    } else {
        // ★ forEachの引数に index を追加します
        data.forEach((item, index) => {
            const tr = document.createElement('tr');
            
            // ★ No. 列の作成
            const tdNo = document.createElement('td');
            tdNo.style.color = '#667eea';
            tdNo.style.fontWeight = '700';
            tdNo.style.fontSize = '0.85rem';
            tdNo.style.whiteSpace = 'nowrap';
            
            // ★ 上から順（最新順）に FB1, FB2, FB3... と番号を振る
            tdNo.textContent = `FB${index + 1}`;

            const date = new Date(item.created_at).toLocaleString('ja-JP', { dateStyle: 'short', timeStyle: 'short' });
            const ratingBadge = item.rating === 'good'
                ? '<span class="badge good">👍 Good</span>'
                : '<span class="badge bad">👎 Bad</span>';

            const tdDate   = document.createElement('td');
            tdDate.style.color = '#999';
            tdDate.style.fontSize = '0.82rem';
            tdDate.style.whiteSpace = 'nowrap';
            tdDate.textContent = date;

            const tdRating = document.createElement('td');
            tdRating.innerHTML = ratingBadge; 

            const tdComment = document.createElement('td');
            tdComment.textContent = item.comment || item.content || '（コメントなし）';

            // ★ tdNo を一番最初に追加
            tr.appendChild(tdNo);
            tr.appendChild(tdDate);
            tr.appendChild(tdRating);
            tr.appendChild(tdComment);
            tbody.appendChild(tr);
        });
    }

    renderChart(good, bad);
}

function renderChart(good, bad) {
    const ctx = document.getElementById('ratingChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Good（高評価）', 'Bad（低評価）'],
            datasets: [{
                data: [good, bad],
                backgroundColor: ['#28a745', '#dc3545'],
                borderWidth: 0,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'bottom', labels: { font: { size: 12 } } } },
            cutout: '65%',
        }
    });
}

// ==================================================
// 2. ベクトル化ステータス
// ==================================================
async function fetchVecStatus() {
    try {
        const res = await fetch('/api/admin/stats/vectorize/status');
        if (!res.ok) throw new Error(res.statusText);
        const d = await res.json();

        // 不要になったチャットログ関連のDOM更新処理を削除済

        const cm = d.anonymous_comments;
        document.getElementById('vecCmNum').textContent = `${cm.vectorized} / ${cm.total}`;
        
        const cmPct = cm.total > 0 ? Math.round(cm.vectorized / cm.total * 100) : 0;
        document.getElementById('vecCmBar').style.width = cmPct + '%';

    } catch (e) {
        console.warn('vectorize status error:', e);
    }
}

// ==================================================
// 3. 一括ベクトル化
// ==================================================
async function runVectorize(target) {
    const btn = document.getElementById('vecBothBtn');
    btn.disabled = true;
    btn.textContent = '処理中...';
    showToast('ベクトル化を開始しました。しばらくお待ちください。', 'info', 8000);

    try {
        const res = await fetch('/api/admin/stats/vectorize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target, limit: 200 }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();

        // 結果サマリーをToastで表示
        let msg = '✅ ベクトル化完了 | ';
        if (data.results.chat_logs) {
            const r = data.results.chat_logs;
            msg += `ログ: ${r.succeeded}件成功 `;
        }
        if (data.results.anonymous_comments) {
            const r = data.results.anonymous_comments;
            msg += `意見: ${r.succeeded}件成功`;
        }
        showToast(msg, 'success', 5000);
        await fetchVecStatus();
    } catch (e) {
        showToast('ベクトル化エラー: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '⚡ 未処理を一括ベクトル化';
    }
}

// ==================================================
// 4. AI相談チャット
// ==================================================
function setQuery(text) {
    document.getElementById('queryInput').value = text;
    document.getElementById('queryInput').focus();
}

function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
}

async function sendQuery() {
    const input   = document.getElementById('queryInput');
    const btn     = document.getElementById('sendBtn');
    const history = document.getElementById('chatHistory');

    // ── 停止ロジック ──
    if (abortController) {
        abortController.abort();
        abortController = null;
        btn.classList.remove('generating');
        btn.title = '送信';
        btn.textContent = '➤';

        // 最後のAIバブルに「停止」を追記
        const bubbles = history.querySelectorAll('.msg.ai');
        if (bubbles.length > 0) {
            const last = bubbles[bubbles.length - 1];
            if (!last.dataset.finished) {
                const stop = document.createElement('span');
                stop.style.cssText = 'display:block;color:#dc3545;font-size:0.78rem;margin-top:6px;font-weight:700;';
                stop.textContent = '⏹ 停止しました';
                last.appendChild(stop);
                last.dataset.finished = 'true';
            }
        }
        return;
    }

    // ── 送信ロジック ──
    const query = input.value.trim();
    if (!query) return;

    appendMsg('user', query);
    input.value = '';
    btn.disabled = false; // abort可能なので disabled にしない
    btn.classList.add('generating');
    btn.title = '停止';
    btn.textContent = '⏹';

    abortController = new AbortController();
    const signal = abortController.signal;

    // AIローディングバブル
    const aiEl = document.createElement('div');
    aiEl.className = 'msg ai';
    aiEl.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    history.appendChild(aiEl);
    history.scrollTop = history.scrollHeight;

    let aiText  = '';
    let isFirst = true;

    try {
        const res = await fetch('/api/admin/stats/chat_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, target_period_days: 30 }),
            signal,
        });

        if (!res.ok) throw new Error(`API Error: ${res.statusText}`);

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            aiText += decoder.decode(value, { stream: true });

            if (isFirst) {
                aiEl.innerHTML = '';
                isFirst = false;
            }

            const rawHtml  = marked.parse(aiText);
            const safeHtml = DOMPurify.sanitize(rawHtml, {
                ALLOWED_TAGS: ['p','br','strong','em','ul','ol','li','h1','h2','h3','h4',
                               'blockquote','code','pre','hr','table','thead','tbody','tr','th','td','a'],
                ALLOWED_ATTR: ['href','target','rel'],
            });
            aiEl.innerHTML = safeHtml;
            history.scrollTop = history.scrollHeight;
        }

        if (isFirst) {
            aiEl.textContent = '（回答を生成できませんでした）';
        }
        aiEl.dataset.finished = 'true';

    } catch (e) {
        if (e.name === 'AbortError') {
            // 停止ロジック側で処理済み
        } else {
            console.error(e);
            aiEl.innerHTML = '';
            const errSpan = document.createElement('span');
            errSpan.style.color = '#dc3545';
            errSpan.textContent = 'エラー: ' + e.message;
            aiEl.appendChild(errSpan);
        }
    } finally {
        if (abortController) {
            abortController = null;
            btn.classList.remove('generating');
            btn.title = '送信';
            btn.textContent = '➤';
        }
    }
}

function appendMsg(role, text) {
    const history = document.getElementById('chatHistory');
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text; // ユーザー入力はテキストとして安全に表示
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

// ==================================================
// 初期化
// ==================================================
(async () => {
    await fetchData();
    await fetchVecStatus();
})();