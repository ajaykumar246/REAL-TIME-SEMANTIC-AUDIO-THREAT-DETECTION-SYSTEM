document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('audio-file');
    const fileNameDisplay = document.getElementById('file-name');
    const startBtn = document.getElementById('start-btn');
    const resetBtn = document.getElementById('reset-btn');
    const globalStatus = document.getElementById('global-status');
    const logContent = document.getElementById('log-content');
    const livePulse = document.getElementById('live-pulse');
    const segmentsGrid = document.getElementById('segments-grid');

    // Verdict Elements
    const verdictBox = document.getElementById('verdict-box');
    const verdictTitle = document.getElementById('verdict-title');
    const statTotal = document.getElementById('stat-total');
    const statSpam = document.getElementById('stat-spam');
    const statHam = document.getElementById('stat-ham');

    // Flow Steps
    const stepStream = document.getElementById('step-stream');
    const stepVad = document.getElementById('step-vad');
    const stepStt = document.getElementById('step-stt');
    const stepMl = document.getElementById('step-ml');

    let currentFileId = null;
    let ws = null;

    // --- File Upload Handling ---
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    async function handleFile(file) {
        if (!file.type.startsWith('audio/')) {
            setStatus('Error: Please upload an audio file.', 'error');
            return;
        }

        fileNameDisplay.textContent = file.name;
        fileNameDisplay.classList.remove('hidden');
        startBtn.disabled = true;
        setStatus('Uploading...', 'status');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.file_id) {
                currentFileId = data.file_id;
                setStatus(`Uploaded: ${data.filename} (${data.size_kb} KB)`, 'success');
                startBtn.disabled = false;
            } else {
                throw new Error('Upload failed');
            }
        } catch (err) {
            console.error(err);
            setStatus('Error uploading file.', 'error');
        }
    }

    // --- UI Helpers ---
    function setStatus(msg, type = 'status') {
        globalStatus.textContent = msg;
        if (type === 'error') globalStatus.style.color = 'var(--warning)';
        else if (type === 'success') globalStatus.style.color = 'var(--success)';
        else globalStatus.style.color = 'var(--text-muted)';
    }

    function appendLog(html, className = '') {
        const div = document.createElement('div');
        div.className = `log-line ${className}`;

        const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        div.innerHTML = `<span style="color: #64748b">[${time}]</span> ${html}`;

        logContent.appendChild(div);
        logContent.scrollTop = logContent.scrollHeight;
    }

    function setActiveStep(stepId) {
        [stepStream, stepVad, stepStt, stepMl].forEach(el => el.classList.remove('active'));
        if (stepId) {
            const step = document.getElementById(`step-${stepId}`);
            if (step) step.classList.add('active');
        }
    }

    // --- WebSocket / Pipeline Execution ---
    startBtn.addEventListener('click', () => {
        if (!currentFileId) return;

        // Reset UI for new run
        logContent.innerHTML = '';
        segmentsGrid.innerHTML = '';
        verdictBox.classList.add('hidden');
        verdictBox.className = 'verdict-box hidden';
        statTotal.textContent = '-';
        statSpam.textContent = '-';
        statHam.textContent = '-';

        startBtn.disabled = true;
        livePulse.classList.remove('hidden');
        appendLog('Initializing WebSocket connection...', 'log-status');

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws/process`);

        ws.onopen = () => {
            appendLog('Connected to backend. Starting pipeline.', 'log-status');
            ws.send(JSON.stringify({ file_id: currentFileId }));
            setActiveStep('stream');
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            switch (msg.type) {
                case 'status':
                    appendLog(msg.message, 'log-status');
                    setStatus(msg.message);
                    break;

                case 'error':
                    appendLog(`Error: ${msg.message}`, 'log-error');
                    setStatus('Pipeline Error', 'error');
                    cleanupRun();
                    break;

                case 'vad':
                    setActiveStep('vad');
                    appendLog(`VAD | Buffer flushed: Segment ${msg.segment} (${msg.duration_s}s speech)`, 'log-vad');
                    break;

                case 'stt':
                    setActiveStep('stt');
                    appendLog(`STT | Seg ${msg.segment} (${msg.language}): "${msg.text}"`, 'log-stt');
                    break;

                case 'classify':
                    setActiveStep('ml');
                    const isSpam = msg.label === 'Spam';
                    const colorClass = isSpam ? 'log-error' : 'log-status';
                    appendLog(`CLS | Seg ${msg.segment} → ${msg.label} (${(msg.confidence * 100).toFixed(0)}%)`, colorClass);

                    // Add segment card to UI
                    renderSegment(msg);
                    break;

                case 'result':
                    setActiveStep(null);
                    showFinalVerdict(msg);
                    cleanupRun();
                    break;
            }
        };

        ws.onerror = () => {
            appendLog('WebSocket Error', 'log-error');
            cleanupRun();
        };

        ws.onclose = () => {
            appendLog('Pipeline finished / Connection closed.', 'log-status');
            cleanupRun();
        };
    });

    function renderSegment(data) {
        const isSpam = data.label === 'Spam';
        const card = document.createElement('div');
        card.className = 'segment-card';

        // ONLY show the specific spam sentence (hide the rest of the conversation)
        // If it's a Ham call, hide the content completely for privacy.
        let displayText;
        if (isSpam) {
            displayText = data.spam_sentence ? `"${data.spam_sentence}"` : `"${data.text}"`;
        } else {
            displayText = `<span style="color: var(--text-muted); font-style: italic;"><i class="fa-solid fa-lock"></i> Safe content (hidden)</span>`;
        }

        card.innerHTML = `
            <div class="seg-meta">
                <span class="seg-number">Segment ${data.segment}</span>
                <span class="seg-label ${isSpam ? 'spam' : 'ham'}">${data.label}</span>
                <span class="seg-conf">${(data.confidence * 100).toFixed(0)}% conf.</span>
            </div>
            <div class="seg-content">
                <div class="seg-text">${displayText}</div>
                <div class="seg-lang">Lang: ${data.language} | Duration: ${data.duration_s}s</div>
            </div>
        `;
        segmentsGrid.appendChild(card);
    }

    function showFinalVerdict(data) {
        verdictBox.classList.remove('hidden');

        statTotal.textContent = data.total_segments;
        statSpam.textContent = data.spam_segments;
        statHam.textContent = data.ham_segments;

        if (data.verdict === 'SPAM') {
            verdictBox.classList.add('spam');
            verdictTitle.innerHTML = '🚨 SPAM CALL DETECTED';
            setStatus('Finished: Spam Detected', 'error');
        } else if (data.verdict === 'HAM') {
            verdictBox.classList.add('ham');
            verdictTitle.innerHTML = '✅ LEGITIMATE CALL (HAM)';
            setStatus('Finished: Legitimate Call', 'success');
        } else {
            verdictTitle.innerHTML = '⚠️ UNKNOWN / NO SPEECH';
            setStatus('Finished: Unknown');
        }
    }

    function cleanupRun() {
        startBtn.disabled = false;
        livePulse.classList.add('hidden');
        setActiveStep(null);
        if (ws) {
            ws.close();
            ws = null;
        }
    }

    // --- Reset ---
    resetBtn.addEventListener('click', () => {
        if (ws) ws.close();
        currentFileId = null;
        fileInput.value = '';
        fileNameDisplay.textContent = '';
        fileNameDisplay.classList.add('hidden');
        startBtn.disabled = true;
        logContent.innerHTML = '';
        segmentsGrid.innerHTML = '';
        verdictBox.className = 'verdict-box hidden';
        setStatus('Ready for new file.');
        setActiveStep(null);
        livePulse.classList.add('hidden');
    });
});
