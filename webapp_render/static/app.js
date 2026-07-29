document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analyze-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    
    const welcomeState = document.getElementById('welcome-state');
    const resultsPanel = document.getElementById('results-panel');
    const resTitle = document.getElementById('res-title');
    const resContext = document.getElementById('res-context');

    
    const errorToast = document.getElementById('error-message');

    function showError(msg) {
        errorToast.textContent = msg;
        errorToast.classList.remove('hidden');
        setTimeout(() => {
            errorToast.classList.add('hidden');
        }, 5000);
    }

    // Tab Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    let activeMode = 'url-mode';

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active content
            activeMode = btn.dataset.tab;
            tabContents.forEach(content => {
                if(content.id === activeMode) {
                    content.classList.remove('hidden');
                } else {
                    content.classList.add('hidden');
                }
            });
        });
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        let payload = {};
        if (activeMode === 'url-mode') {
            payload.url = document.getElementById('url').value;
        } else {
            payload.raw_text = document.getElementById('raw_text').value;
            payload.outlet = document.getElementById('outlet').value;
        }
        payload.entity = document.getElementById('entity').value;
        
        // UI Loading State
        submitBtn.disabled = true;
        btnText.textContent = 'Executing...';
        loader.classList.remove('hidden');
        
        // Hide welcome state, show results panel (but maybe dim it)
        welcomeState.classList.add('hidden');
        resultsPanel.classList.add('hidden');
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to analyze article');
            }
            
            // Update Engine Info
            const engineDisplay = document.getElementById('engine-display');
            if (engineDisplay && data.engine) {
                engineDisplay.textContent = data.engine;
            }
            
            // Populate Target Entity Header & Title
            const targetEnt = data.target_entity || 'Primary Entity';
            let titleSub = `<span style="display: inline-block; margin-top: 0.4rem; padding: 0.2rem 0.6rem; border-radius: 4px; background: rgba(59,130,246,0.15); color: #60a5fa; font-family: monospace; font-size: 0.9rem; border: 1px solid rgba(59,130,246,0.3);">Framing Target: <strong>${targetEnt}</strong></span>`;
            if (data.auto_entity) {
                titleSub += ` <span style="font-size: 0.8rem; color: #10b981; font-family: monospace;">(Auto-Detected Target)</span>`;
            }
            resTitle.innerHTML = `${data.title} <br>${titleSub}`;
            resContext.textContent = `"...${data.context}..."`;
            
            const bartTitle = document.getElementById('bart-card-title');
            if (bartTitle) bartTitle.textContent = `Framing Distribution for Target: "${targetEnt}"`;
            
            // Clear old bars
            const barsContainer = document.getElementById('bart-bars-container');
            if (barsContainer) barsContainer.innerHTML = '';
            
            function renderScores(scoresObj, container) {
                if (!scoresObj || !container) {
                    if (container) container.innerHTML = '<p style="color: #64748b; font-size: 0.9rem;">Model unavailable.</p>';
                    return;
                }
                const sortedScores = Object.entries(scoresObj).sort((a, b) => b[1] - a[1]);
                sortedScores.forEach(([label, score]) => {
                    const percentage = (score * 100).toFixed(1);
                    const barRow = document.createElement('div');
                    barRow.className = 'bar-row';
                    barRow.innerHTML = `
                        <div class="bar-labels">
                            <span class="label-name">${label}</span>
                            <span class="label-score">${percentage}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill fill-${label.replace(/\s+/g, '-')}"></div>
                        </div>
                    `;
                    container.appendChild(barRow);
                    setTimeout(() => {
                        const fillElement = barRow.querySelector('.bar-fill');
                        if (fillElement) fillElement.style.width = `${percentage}%`;
                    }, 50);
                });
            }
            
            renderScores(data.scores || data.bart_scores, barsContainer);

            // Render Cross-Entity Framing Comparison
            const multiCard = document.getElementById('multi-entity-card');
            const multiContainer = document.getElementById('multi-entity-container');
            if (multiCard && multiContainer) {
                multiContainer.innerHTML = '';
                if (data.other_entity_scores && Object.keys(data.other_entity_scores).length > 0) {
                    multiCard.classList.remove('hidden');
                    Object.entries(data.other_entity_scores).forEach(([entName, scoresObj]) => {
                        const entBox = document.createElement('div');
                        entBox.style.cssText = 'background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 0.8rem;';
                        
                        const sorted = Object.entries(scoresObj).sort((a, b) => b[1] - a[1]);
                        let rowsHtml = sorted.map(([lbl, val]) => `
                            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;">
                                <span>${lbl}</span>
                                <span style="color: #cbd5e1; font-family: monospace;">${(val * 100).toFixed(1)}%</span>
                            </div>
                        `).join('');
                        
                        entBox.innerHTML = `
                            <div style="font-weight: 600; color: #e2e8f0; font-size: 0.9rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.4rem; margin-bottom: 0.4rem;">
                                Entity: <span style="color: #38bdf8;">${entName}</span>
                            </div>
                            ${rowsHtml}
                        `;
                        multiContainer.appendChild(entBox);
                    });
                } else {
                    multiCard.classList.add('hidden');
                }
            }
            
            resultsPanel.classList.remove('hidden');
            
        } catch (err) {
            showError(err.message);
            // If it failed and we haven't shown results before, show welcome state again
            if (resTitle.textContent === '...') {
                welcomeState.classList.remove('hidden');
            }
        } finally {
            // Restore UI
            submitBtn.disabled = false;
            btnText.textContent = 'Execute Analysis';
            loader.classList.add('hidden');
        }
    });
});
