// =========================
// CẤU HÌNH
// =========================
const API_BASE_URL = 'http://localhost:5000';

// =========================
// DOM ELEMENTS
// =========================
const container = document.querySelector('.container');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const emailPreview = document.getElementById('emailPreview');
const previewSubject = document.getElementById('previewSubject');
const previewBody = document.getElementById('previewBody');
const modelsContainer = document.getElementById('modelsContainer');
const summaryCard = document.getElementById('summaryCard');
const summaryResult = document.getElementById('summaryResult');
const summaryModels = document.getElementById('summaryModels');
const errorMessage = document.getElementById('errorMessage');
const xaiModal = document.getElementById('xaiModal');
const modalClose = document.getElementById('modalClose');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const xaiDetailModeCheckbox = document.getElementById('xaiDetailMode');

// =========================
// STATE
// =========================
let currentEmail = null;
let currentResults = null;
// Cache kết quả giải thích để tránh gọi API lặp lại
// Key: `${modelName}::${emailText.slice(0, 1000)}`
const explanationCache = new Map();

// =========================
// INIT
// =========================
document.addEventListener('DOMContentLoaded', async () => {
    await loadEmailFromStorage();
    setupEventListeners();
});

// =========================
// EVENT LISTENERS
// =========================
function setupEventListeners() {
    analyzeBtn.addEventListener('click', handleAnalyze);
    modalClose.addEventListener('click', closeModal);

    xaiModal.addEventListener('click', (e) => {
        if (e.target === xaiModal) closeModal();
    });

    chrome.runtime.onMessage.addListener((message) => {
        if (message.type === 'emailUpdated') {
            loadEmailFromStorage();
        }
    });
}

// =========================
// LOAD EMAIL
// =========================
async function loadEmailFromStorage() {
    try {
        const result = await chrome.storage.local.get(['currentEmail']);
        if (result.currentEmail) {
            currentEmail = result.currentEmail;
            // Mỗi lần email thay đổi thì xoá cache explanation cũ
            explanationCache.clear();
            updateEmailPreview(currentEmail);
            analyzeBtn.disabled = false;
        } else {
            analyzeBtn.disabled = true;
        }
    } catch (error) {
        console.error('Lỗi khi tải email:', error);
        showError('Không thể tải email từ storage');
    }
}

// =========================
// UPDATE PREVIEW (RESET HIGHLIGHT)
// =========================
function updateEmailPreview(email) {
    previewSubject.textContent = email.subject || 'Không có tiêu đề';
    previewBody.textContent = email.body || 'Không có nội dung';
}

// =========================
// ANALYZE EMAIL
// =========================
async function handleAnalyze() {
    if (!currentEmail) {
        showError('Không có email để phân tích');
        return;
    }

    analyzeBtn.disabled = true;
    loading.style.display = 'flex';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';

    try {
        const emailText = `${currentEmail.subject || ''} ${currentEmail.body || ''}`.trim();

        // ---- Predict
        const resp = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email_text: emailText })
        });

        if (!resp.ok) {
            throw new Error(`HTTP ${resp.status}`);
        }

        const data = await resp.json();
        currentResults = data;

        displayResults(data, 'unknown');

        // ---- Highlight theo BERT (nếu có)
        const hasBert = !!data.predictions?.BERT;

        if (hasBert) {
            try {
                await highlightPreviewWithBert(emailText);
            } catch (e) {
                console.warn('Highlight BERT failed:', e);
            }
        }

    } catch (error) {
        console.error('Analyze error:', error);
        showError(`Lỗi phân tích: ${error.message}`);
    } finally {
        analyzeBtn.disabled = false;
        loading.style.display = 'none';
    }
}

// =========================
// DISPLAY RESULTS
// =========================
function displayResults(data, detectedLanguage = 'unknown') {
    resultsSection.style.display = 'block';

    const predictions = data.predictions || {};
    const modelNames = Object.keys(predictions);

    let phishingCount = 0;
    let benignCount = 0;

    modelNames.forEach(m => {
        if (predictions[m].label === 'phishing') phishingCount++;
        else benignCount++;
    });

    const consensus = phishingCount > benignCount ? 'PHISHING' : 'BENIGN';
    summaryResult.textContent = consensus;
    summaryResult.className = `summary-value ${consensus.toLowerCase()}`;

    const langText = detectedLanguage === 'vi'
        ? ' (Tiếng Việt)'
        : detectedLanguage === 'en'
        ? ' (Tiếng Anh)'
        : '';

    summaryModels.textContent =
        consensus === 'PHISHING'
            ? `${phishingCount}/${modelNames.length} models phát hiện phishing${langText}`
            : `${benignCount}/${modelNames.length} models phát hiện benign${langText}`;

    summaryCard.className = `summary-card ${consensus.toLowerCase()}`;

    if (consensus === 'PHISHING') {
        showWarning('Cảnh báo: Đa số models phát hiện email này là PHISHING!');
    }

    modelsContainer.innerHTML = '';
    modelNames.forEach(m => {
        modelsContainer.appendChild(createModelCard(m, predictions[m]));
    });
}

// =========================
// MODEL CARD
// =========================
function createModelCard(modelName, prediction) {
    const card = document.createElement('div');
    card.className = `model-card ${prediction.label}`;

    const labelClass = prediction.label === 'phishing' ? 'phishing' : 'benign';
    const confidence = Math.round(prediction.probability * 100);

    card.innerHTML = `
        <div class="model-header">
            <span class="model-name">${modelName}</span>
            <span class="model-label ${labelClass}">${prediction.label.toUpperCase()}</span>
        </div>
        <div class="confidence-section">
            <div class="confidence-label">
                <span>Độ tin cậy:</span>
                <span class="confidence-value">${confidence}%</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar ${labelClass}" style="width:${confidence}%"></div>
            </div>
        </div>
        <button class="explain-btn">🔍 Giải thích dự đoán</button>
    `;

    card.querySelector('.explain-btn').addEventListener('click', () => {
        showXAIExplanation(modelName, prediction);
    });

    return card;
}

// =========================
// XAI MODAL
// =========================
function makeExplainKey(modelName, emailText) {
    return `${modelName}::${(emailText || '').slice(0, 1000)}`;
}

async function fetchExplanation(modelName, emailText, mode = 'quick') {
    const key = `${makeExplainKey(modelName, emailText)}::${mode}`;

    if (explanationCache.has(key)) {
        return explanationCache.get(key);
    }

    const resp = await fetch(`${API_BASE_URL}/api/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model_name: modelName,
            email_text: emailText,
            mode
        })
    });

    if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
    }

    const data = await resp.json();
    explanationCache.set(key, data);
    return data;
}

async function showXAIExplanation(modelName, prediction) {
    if (!currentEmail) return;

    modalTitle.textContent = `Giải thích dự đoán - ${modelName}`;
    modalBody.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    xaiModal.classList.add('show');

    try {
        const mode = xaiDetailModeCheckbox?.checked ? 'full' : 'quick';
        const data = await fetchExplanation(
            modelName,
            `${currentEmail.subject || ''} ${currentEmail.body || ''}`.trim(),
            mode
        );
        displayXAIContent(data, prediction);

    } catch (e) {
        modalBody.innerHTML = `<div class="error-message">${e.message}</div>`;
    }
}

function displayXAIContent(data, prediction) {
    const confidence = Math.round(prediction.probability * 100);
    const labelUpper = prediction.label.toUpperCase();

    const tokens = Array.isArray(data.important_tokens) ? data.important_tokens : [];
    const positiveTokens = tokens.filter(t => t.weight > 0);
    const negativeTokens = tokens.filter(t => t.weight < 0);

    const positiveTop = positiveTokens.slice(0, 3).map(t => `"${t.token}"`).join(', ');
    const negativeTop = negativeTokens.slice(0, 3).map(t => `"${t.token}"`).join(', ');

    let summaryText = '';
    if (labelUpper === 'PHISHING') {
        summaryText = `
            Mô hình đánh giá email này là <strong class="xai-prediction-label phishing">PHISHING (${confidence}%)</strong>
            vì xuất hiện nhiều từ/cụm từ mang tính rủi ro như ${positiveTop || '...'}
            và thiếu các từ thể hiện ngữ cảnh bình thường như ${negativeTop || '...'}.
        `;
    } else {
        summaryText = `
            Mô hình đánh giá email này là <strong class="xai-prediction-label benign">BENIGN (${confidence}%)</strong>
            vì xuất hiện nhiều từ liên quan đến ngữ cảnh công việc/bình thường như ${negativeTop || '...'}
            và ít dấu hiệu rủi ro từ các từ như ${positiveTop || '...'}.
        `;
    }

    const renderTokenChip = (t) => `
        <div class="token-item ${t.weight > 0 ? 'positive' : 'negative'}">
            <span class="token-text">${t.token}</span>
            <span class="token-weight">${t.weight.toFixed(3)}</span>
        </div>
    `;

    const positiveListHTML = positiveTokens.length
        ? `<div class="xai-tokens-group">
                <div class="xai-tokens-title phishing">Từ/cụm từ làm tăng khả năng PHISHING:</div>
                <div class="token-list">
                    ${positiveTokens.map(renderTokenChip).join('')}
                </div>
           </div>`
        : '';

    const negativeListHTML = negativeTokens.length
        ? `<div class="xai-tokens-group">
                <div class="xai-tokens-title benign">Từ/cụm từ làm tăng khả năng BENIGN:</div>
                <div class="token-list">
                    ${negativeTokens.map(renderTokenChip).join('')}
                </div>
           </div>`
        : '';

    modalBody.innerHTML = `
        <div class="xai-content">
            <div class="xai-prediction">
                <div class="xai-prediction-summary">
                    ${summaryText}
                </div>
            </div>
            <div class="xai-tokens">
                ${positiveListHTML}
                ${negativeListHTML}
            </div>
        </div>
    `;
}

// =========================
// HIGHLIGHT LOGIC (SHAP)
// =========================
function escapeHTML(str) {
    return (str || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function escapeRegExp(str) {
    return (str || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getHighlightStyle(weight) {
    const abs = Math.min(Math.abs(weight), 1);
    const alpha = 0.15 + 0.55 * abs;
    return weight > 0
        ? `background:rgba(231,76,60,${alpha});`
        : `background:rgba(39,174,96,${alpha});`;
}

function highlightEmailText(rawText, tokens) {
    let text = escapeHTML(rawText);
    if (!tokens?.length) return text;

    // Các token không nên được highlight vì có thể trùng với tên thuộc tính HTML
    // hoặc tag, dễ làm hỏng markup khi thay thế nhiều lần.
    const forbiddenTokens = new Set([
        'style',
        'class',
        'span',
        'div',
        'background',
        'rgba',
        'color',
        'width',
        'height'
    ]);

    tokens
        .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
        .forEach(t => {
            if (!t.token || t.token.length < 2) return;

            const tokenLower = t.token.toLowerCase();
            if (forbiddenTokens.has(tokenLower)) {
                return;
            }

            const regex = new RegExp(`\\b(${escapeRegExp(t.token)})\\b`, 'gi');
            text = text.replace(
                regex,
                `<span class="highlight-token" style="${getHighlightStyle(t.weight)}">$1</span>`
            );
        });

    return text;
}

async function highlightPreviewWithBert(emailText, detectedLanguage) {
    const modelName = 'BERT';

    // Highlight luôn dùng chế độ nhanh để đảm bảo phản hồi tốt
    const data = await fetchExplanation(modelName, emailText, 'quick');
    previewBody.innerHTML = highlightEmailText(currentEmail.body, data.important_tokens);
}

// =========================
// UI HELPERS
// =========================
function closeModal() {
    xaiModal.classList.remove('show');
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function showWarning(message) {
    let w = document.getElementById('warningMessage');
    if (!w) {
        w = document.createElement('div');
        w.id = 'warningMessage';
        w.className = 'warning-message';
        container.insertBefore(w, resultsSection);
    }
    w.textContent = message;
    w.style.display = 'block';
    setTimeout(() => (w.style.display = 'none'), 10000);
}
