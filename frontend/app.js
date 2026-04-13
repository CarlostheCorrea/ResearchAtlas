/* ArXiv Research Assistant — frontend logic
 * All API calls use fetch() with async/await.
 * Debug logging: set DEBUG = true in browser console to enable.
 */

const DEBUG = false;
function debug(...args) { if (DEBUG) console.log('[ResearchAtlas]', ...args); }

// ── Theme ──────────────────────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('ra-theme') || 'light';
  document.documentElement.dataset.theme = saved;
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme || 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('ra-theme', next);
}

// ── State ──────────────────────────────────────────────────────────────────────
const state = {
  currentSession: null,
  currentPaper: null,
  searchResults: [],
  shortlist: [],
  library: [],        // kept in sync with DB; used for in-search indicators
  libraryIds: new Set(), // fast O(1) lookup for "is this paper saved?"
  pollingInterval: null,
  pendingInterrupt: null,
  currentQaResult: null,
  qaThreadSessionId: null,
  qaThreadPaperId: null,
  currentQaPoll: null,
  availableQaTools: [],
  pdfDoc: null,
  pdfUrl: null,
  pdfCurrentPage: 1,
  pdfScale: 1.2,
  pdfEvidenceIndex: null,
  qaLogCount: 0,
  searchMode: 'topic',      // 'topic' | 'author'
  coauthorGraph: null,      // graph data from API
  activeAuthorFilter: null, // normalized author id being filtered
  _d3Simulation: null,      // active D3 simulation (stopped on tab switch)
};

// ── API helpers ────────────────────────────────────────────────────────────────
async function apiPost(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

async function apiDelete(path) {
  const res = await fetch(path, { method: 'DELETE' });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json();
}

// ── Startup: load library into landing page ───────────────────────────────────
async function initApp() {
  try {
    const papers = await apiGet('/api/library');
    state.library = papers || [];
    state.libraryIds = new Set(state.library.map(p => p.arxiv_id));
    renderLandingLibrary(state.library);
  } catch (e) {
    // Library may be empty on first run — that's fine
  }
}

function renderLandingLibrary(papers) {
  const emptyState = document.getElementById('empty-state');
  if (!papers || papers.length === 0) {
    // Default empty state is already shown
    return;
  }

  emptyState.innerHTML = `
    <div style="width:100%;text-align:left;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:20px;">
        <h2 style="font-size:18px;font-weight:600;color:var(--text);margin:0;">Your Research Library</h2>
        <span style="font-size:12px;color:var(--text-muted)">${papers.length} saved paper${papers.length !== 1 ? 's' : ''}</span>
      </div>
      <div style="display:grid;gap:10px;">
        ${papers.map(p => `
          <div class="landing-lib-card" data-arxiv="${p.arxiv_id}" style="
            background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
            padding:14px 16px;cursor:pointer;transition:border-color 0.15s ease,transform 0.15s ease;
          ">
            <div style="font-size:14px;font-weight:500;color:var(--text);margin-bottom:4px;line-height:1.4;">
              ${escHtml(p.title || p.arxiv_id)}
            </div>
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
              <span style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted)">${p.arxiv_id}</span>
              <span style="font-size:11px;color:var(--text-muted)">Saved ${formatDate(p.saved_at)}</span>
              ${p.rating ? `<span style="color:var(--amber);font-size:12px;">${'★'.repeat(p.rating)}${'☆'.repeat(5 - p.rating)}</span>` : ''}
              <span style="font-family:var(--font-mono);font-size:10px;padding:1px 7px;background:rgba(108,99,255,0.15);
                color:var(--accent);border:1px solid var(--accent-dim);border-radius:4px;">In Library</span>
            </div>
          </div>
        `).join('')}
      </div>
      <div style="margin-top:16px;text-align:center;color:var(--text-muted);font-size:13px;">
        Search for more papers above, or click a saved paper to view its summary.
      </div>
    </div>
  `;

  // Clicking a saved paper opens its summary directly
  emptyState.querySelectorAll('.landing-lib-card').forEach(card => {
    card.addEventListener('mouseenter', () => { card.style.borderColor = 'var(--accent)'; card.style.transform = 'translateY(-1px)'; });
    card.addEventListener('mouseleave', () => { card.style.borderColor = 'var(--border)'; card.style.transform = ''; });
    card.addEventListener('click', () => {
      const arxivId = card.dataset.arxiv;
      const saved = state.library.find(p => p.arxiv_id === arxivId);
      if (!saved) return;
      // Build a minimal paper object from the saved record and open it
      openSavedPaper(saved);
    });
  });
}

function openSavedPaper(savedPaper) {
  // Construct a paper-like object from the saved record plus cached arXiv metadata.
  const paper = {
    arxiv_id: savedPaper.arxiv_id,
    title: savedPaper.title || savedPaper.arxiv_id,
    authors: Array.isArray(savedPaper.authors) ? savedPaper.authors : [],
    abstract: savedPaper.abstract || '',
    published: savedPaper.published || '',
    categories: Array.isArray(savedPaper.categories) ? savedPaper.categories : [],
    pdf_url: savedPaper.pdf_url || `https://arxiv.org/pdf/${savedPaper.arxiv_id}`,
  };

  // Try to parse the saved summary JSON
  let summary = null;
  try {
    summary = savedPaper.summary_json ? JSON.parse(savedPaper.summary_json) : null;
  } catch (e) {}

  state.currentPaper = paper;
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('paper-view').classList.remove('hidden');
  renderPaperHeader(paper, null);
  document.getElementById('badge-saved')?.classList.remove('hidden');
  document.getElementById('badge-analyzed')?.classList.remove('hidden');

  switchTab('summary');
  if (summary) {
    renderSummary(summary);
  } else {
    document.getElementById('tab-summary').innerHTML =
      '<div style="padding:16px;color:var(--text-muted);font-size:13px;">Summary not available. Click "Analyze Paper" to generate one.</div>';
  }
  resetQaWorkspace();
}

// ── Search flow ────────────────────────────────────────────────────────────────
async function handleSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) return;

  const yearFrom = document.getElementById('year-from').value;
  const catInput = document.getElementById('categories').value.trim();
  const categories = catInput ? catInput.split(',').map(s => s.trim()).filter(Boolean) : null;

  setLoading(true);
  updateStatusBar('Searching arXiv...', 'running');

  try {
    const result = await apiPost('/api/search', {
      query,
      max_results: 20,
      year_from: yearFrom ? parseInt(yearFrom) : null,
      categories,
      search_mode: state.searchMode,
    });

    state.currentSession = result.session_id;
    state.searchResults = result.ranked_results || [];

    // Show filter report
    renderFilterReport(result.filter_report);

    // Render paper cards
    renderResultsList(state.searchResults);

    // Handle co-author graph for author mode
    if (result.coauthor_graph) {
      state.coauthorGraph = result.coauthor_graph;
      // Show the network tab button
      document.getElementById('tab-network').classList.remove('hidden');
      // Clear any old filter chip
      clearAuthorFilter();
    } else {
      state.coauthorGraph = null;
      document.getElementById('tab-network').classList.add('hidden');
    }

    const count = state.searchResults.length;
    updateStatusBar(`Found ${count} papers`, 'ok');
    showToast(`${count} papers ranked`, 'success');

    // Auto-switch to network tab after author search.
    // Show the paper-view shell (needed to host the tab bar) but the
    // author header replaces the paper header so no stale paper title shows.
    if (result.coauthor_graph && state.searchMode === 'author') {
      document.getElementById('empty-state').classList.add('hidden');
      document.getElementById('paper-view').classList.remove('hidden');
      // Clear any previously selected paper header so it doesn't bleed through
      document.getElementById('paper-header').innerHTML = '';
      switchTab('network');
    }
  } catch (err) {
    updateStatusBar('Search failed', 'error');
    showToast(err.message, 'error');
  } finally {
    setLoading(false);
  }
}

function renderFilterReport(report) {
  const bar = document.getElementById('filter-report-bar');
  if (!report || !report.total_fetched) { bar.textContent = ''; return; }
  bar.innerHTML = `
    <span class="filter-count-total">${report.total_fetched} fetched</span>
    <span style="color:var(--text-muted)">→</span>
    <span class="filter-count-pass">${report.passed} passed</span>
    <span style="color:var(--text-muted)">·</span>
    <span class="filter-count-drop">${report.dropped} filtered</span>
    <span style="color:var(--text-muted)">(${report.pass_rate || ''})</span>
  `;
}

function renderResultsList(rankedPapers) {
  const list = document.getElementById('results-list');
  list.innerHTML = '';

  if (!rankedPapers || rankedPapers.length === 0) {
    list.innerHTML = '<div style="padding:16px;color:var(--text-muted);text-align:center;font-size:13px;">No results found. Try different keywords.</div>';
    return;
  }

  rankedPapers.forEach(rp => {
    const card = renderPaperCard(rp);
    list.appendChild(card);
  });
}

function renderPaperCard(rankedPaper) {
  const paper = rankedPaper.paper || rankedPaper;
  const score = rankedPaper.composite_score || 0;
  const scoreClass = score >= 70 ? 'score-high' : score >= 40 ? 'score-mid' : 'score-low';
  const barClass = score >= 70 ? 'score-bar-green' : score >= 40 ? 'score-bar-amber' : 'score-bar-red';
  const inLibrary = state.libraryIds.has(paper.arxiv_id);

  const authorStr = (paper.authors || []).slice(0, 3).join(', ') +
    (paper.authors && paper.authors.length > 3 ? ' et al.' : '');
  const year = (paper.published || '').slice(0, 4);
  const cats = (paper.categories || []).slice(0, 2);

  const el = document.createElement('div');
  el.className = `paper-card ${scoreClass}${inLibrary ? ' in-library' : ''}`;
  el.dataset.arxivId = paper.arxiv_id;
  el.innerHTML = `
    <button class="card-star-btn" title="Shortlist paper" data-id="${paper.arxiv_id}">☆</button>
    <div class="card-title">${escHtml(paper.title)}</div>
    <div class="card-authors">${escHtml(authorStr)}</div>
    <div class="card-meta">
      <span class="card-year">${year}</span>
      ${cats.map(c => `<span class="card-category">${escHtml(c)}</span>`).join('')}
      ${inLibrary ? `<span class="card-in-library-badge">✓ In Library</span>` : ''}
    </div>
    <div class="card-score-bar-wrap">
      <div class="card-score-bar ${barClass}" style="width:${Math.round(score)}%"></div>
    </div>
  `;

  el.addEventListener('click', (e) => {
    if (e.target.classList.contains('card-star-btn')) return;
    document.querySelectorAll('.paper-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    selectPaper(rankedPaper);
  });

  el.querySelector('.card-star-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    shortlistPaper(paper.arxiv_id, el.querySelector('.card-star-btn'));
  });

  return el;
}

// ── Paper selection ────────────────────────────────────────────────────────────
async function selectPaper(rankedPaper) {
  const paper = rankedPaper.paper || rankedPaper;
  state.currentPaper = paper;

  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('paper-view').classList.remove('hidden');

  renderPaperHeader(paper, rankedPaper);

  // Default to summary tab, clear ALL old content so previous results never bleed through
  switchTab('summary');
  document.getElementById('tab-summary').innerHTML =
    '<div style="color:var(--text-muted);padding:16px;font-size:13px;">Click "Analyze Paper" to generate a summary, or ask a question below.</div>';
  resetQaWorkspace();
  // Stop any in-flight polling from a previous paper
  if (state.pollingInterval) {
    clearInterval(state.pollingInterval);
    state.pollingInterval = null;
  }
}

function renderPaperHeader(paper, rankedPaper) {
  const header = document.getElementById('paper-header');
  const year = (paper.published || '').slice(0, 4);
  const cats = (paper.categories || []).slice(0, 3);
  const authorStr = (paper.authors || []).slice(0, 4).join(', ') +
    (paper.authors && paper.authors.length > 4 ? ' et al.' : '');
  const score = rankedPaper ? rankedPaper.composite_score : null;

  header.innerHTML = `
    <div class="paper-title">${escHtml(paper.title)}</div>
    <div class="paper-authors">${escHtml(authorStr)}</div>
    <div class="paper-badges">
      ${year ? `<span class="badge badge-year">${year}</span>` : ''}
      ${cats.map(c => `<span class="badge badge-cat">${escHtml(c)}</span>`).join('')}
      ${score !== null ? `<span class="badge badge-cat" style="color:var(--text)">Score: ${score.toFixed(1)}</span>` : ''}
      <span class="badge badge-analyzed hidden" id="badge-analyzed">✓ Analyzed</span>
      <span class="badge badge-saved hidden" id="badge-saved">Saved</span>
    </div>
    <div class="paper-abstract" id="paper-abstract" title="Click to expand">${escHtml(paper.abstract)}</div>
    <div class="paper-actions" id="paper-actions">
      <button class="btn-primary" id="btn-analyze">Analyze Paper</button>
      <a href="https://arxiv.org/abs/${paper.arxiv_id}" target="_blank" class="btn-ghost" style="text-decoration:none;display:inline-flex;align-items:center;">View on arXiv ↗</a>
    </div>
  `;

  document.getElementById('paper-abstract').addEventListener('click', function() {
    this.classList.toggle('expanded');
  });
  document.getElementById('btn-analyze').addEventListener('click', () => {
    triggerAnalysis(paper.arxiv_id);
  });
}

// ── Analysis flow ──────────────────────────────────────────────────────────────
async function triggerAnalysis(arxiv_id) {
  // Always use a fresh session ID for analysis so the LangGraph checkpointer
  // starts a clean thread — prevents old Q&A answers from leaking into the summary.
  const session_id = generateId();
  state.currentSession = session_id;

  updateStatusBar('Starting analysis...', 'running');

  try {
    const result = await apiPost('/api/chat', {
      message: 'analyze',
      arxiv_id,
      session_id,
    });

    state.currentSession = result.session_id || session_id;
    startPolling(state.currentSession);
  } catch (err) {
    updateStatusBar('Analysis failed', 'error');
    showToast(err.message, 'error');
  }
}

function startPolling(session_id) {
  if (state.pollingInterval) clearInterval(state.pollingInterval);

  const progressMessages = [
    'Downloading PDF...',
    'Extracting text...',
    'Building index...',
    'Generating competitive summaries...',
  ];
  let msgIdx = 0;

  updateStatusBar(progressMessages[msgIdx], 'running');
  document.getElementById('tab-summary').innerHTML = renderProgressSteps(0);

  state.pollingInterval = setInterval(async () => {
    try {
      const result = await apiGet(`/api/chat/status/${session_id}`);
      debug('poll result:', result);

      if (result.status === 'running') {
        msgIdx = Math.min(msgIdx + 1, progressMessages.length - 1);
        updateStatusBar(progressMessages[msgIdx], 'running');
        document.getElementById('tab-summary').innerHTML = renderProgressSteps(msgIdx);
      }

      if (result.status === 'interrupted') {
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
        state.pendingInterrupt = result;
        showApprovalModal(result.interrupt_payload || {});
      }

      if (result.status === 'completed') {
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
        updateStatusBar('Analysis complete', 'ok');

        if (result.summary) {
          renderSummary(result.summary);
          document.getElementById('badge-analyzed')?.classList.remove('hidden');
        } else if (result.final_answer) {
          renderChatMessage('assistant', result.final_answer, []);
        }
      }

      if (result.status === 'error') {
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
        updateStatusBar('Error', 'error');
        showToast(result.error || 'An error occurred', 'error');
      }
    } catch (err) {
      debug('poll error:', err);
    }
  }, 2000);
}

function renderProgressSteps(activeIdx) {
  const steps = [
    'Downloading PDF',
    'Extracting & cleaning text',
    'Building semantic index',
    'Generating competitive summaries',
  ];
  return `
    <div style="padding:16px;">
      <div style="color:var(--text-muted);font-size:13px;margin-bottom:12px;">Analyzing paper — please wait...</div>
      <div class="progress-steps">
        ${steps.map((s, i) => `
          <div class="progress-step">
            <span class="step-icon ${i < activeIdx ? 'step-done' : i === activeIdx ? 'step-active' : ''}">
              ${i < activeIdx ? '✓' : i === activeIdx ? '⟳' : '○'}
            </span>
            <span style="color:${i === activeIdx ? 'var(--amber)' : i < activeIdx ? 'var(--green)' : 'var(--text-muted)'}">${s}</span>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

// ── Summary rendering ──────────────────────────────────────────────────────────
function renderSummary(summary) {
  const pane = document.getElementById('tab-summary');
  if (!summary) { pane.innerHTML = '<div style="padding:16px;color:var(--text-muted)">No summary available.</div>'; return; }

  const confidenceClass = (summary.confidence_note || '').toLowerCase().includes('full')
    ? 'confidence-full' : 'confidence-partial';
  const confidenceIcon = (summary.confidence_note || '').toLowerCase().includes('full') ? '●' : '◑';

  const sections = [
    { key: 'overview',              label: 'Overview',              open: true },
    { key: 'problem_addressed',     label: 'Problem Addressed' },
    { key: 'main_contribution',     label: 'Main Contribution' },
    { key: 'method',                label: 'Methodology' },
    { key: 'datasets_experiments',  label: 'Datasets & Experiments' },
    { key: 'results',               label: 'Results' },
    { key: 'limitations',           label: 'Limitations' },
    { key: 'why_it_matters',        label: 'Why It Matters' },
  ];

  pane.innerHTML = `
    <div class="${confidenceClass} confidence-badge">${confidenceIcon} ${escHtml(summary.confidence_note || '')}</div>
    ${sections.map(s => `
      <div class="summary-section ${s.open ? 'open' : ''}" data-key="${s.key}">
        <div class="summary-section-header">
          <span class="summary-section-label">${s.label}</span>
          <span class="summary-section-chevron">▶</span>
        </div>
        <div class="summary-section-body">
          <div class="summary-section-text">${escHtml(summary[s.key] || 'Not available.')}</div>
        </div>
      </div>
    `).join('')}
    <div style="margin-top:16px;">
      <button class="btn-primary" id="btn-save-library">Save to Library</button>
    </div>
  `;

  // Toggle collapsible sections
  pane.querySelectorAll('.summary-section-header').forEach(header => {
    header.addEventListener('click', () => {
      header.parentElement.classList.toggle('open');
    });
  });

  document.getElementById('btn-save-library')?.addEventListener('click', () => {
    triggerSave(summary);
  });
}

async function triggerSave(summary) {
  if (!state.currentPaper) return;
  const session_id = state.currentSession || generateId();

  try {
    const result = await apiPost('/api/chat', {
      message: 'save',
      arxiv_id: state.currentPaper.arxiv_id,
      session_id,
    });
    state.currentSession = result.session_id || session_id;
    startPolling(state.currentSession);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Summary evaluation scorecard ───────────────────────────────────────────────
function renderSummaryEvaluation(evaluation) {
  if (!evaluation) return '';

  const metrics = [
    { key: 'faithfulness',    label: 'Faithfulness',     desc: 'Claims grounded in paper text' },
    { key: 'specificity',     label: 'Specificity',      desc: 'Concrete details from the paper' },
    { key: 'completeness',    label: 'Completeness',     desc: 'All 8 sections substantively filled' },
    { key: 'section_accuracy',label: 'Section Accuracy', desc: 'Each section answers its purpose' },
  ];

  const overall = (evaluation.overall_status || 'needs_review').replace(/_/g, ' ');
  const faithFail = evaluation.faithfulness?.status === 'fail';
  const needsReview = evaluation.overall_status === 'needs_review';

  const overallColor = needsReview ? '#b91c1c' : '#2e7d32';
  const warningHtml = faithFail
    ? `<div class="summary-eval-warning">⚠ Some claims may not be grounded in the paper text. Consider requesting a revision.</div>`
    : '';

  const cardsHtml = metrics.map(({ key, label, desc }) => {
    const item = evaluation[key] || {};
    const status = item.status || 'not_applicable';
    const score = typeof item.score === 'number' ? item.score.toFixed(2) : '—';
    const cls = status === 'pass' ? 'pass' : status === 'fail' ? 'fail' : 'na';
    return `
      <div class="summary-eval-card summary-eval-${cls}">
        <div class="summary-eval-head">
          <span>${escHtml(label)}</span>
          <strong>${escHtml(score)}</strong>
        </div>
        <div class="summary-eval-status">${escHtml(status.replace(/_/g, ' '))}</div>
        <div class="summary-eval-note">${escHtml(item.note || desc)}</div>
      </div>`;
  }).join('');

  const grounding = evaluation.grounding_thought;
  const coverage  = evaluation.coverage_thought;
  const thoughtHtml = (grounding || coverage) ? `
    ${grounding ? `<div class="summary-eval-thought"><strong>Grounding:</strong> ${escHtml(grounding)}</div>` : ''}
    ${coverage  ? `<div class="summary-eval-thought"><strong>Coverage:</strong> ${escHtml(coverage)}</div>`  : ''}
  ` : '';

  return `
    <div class="summary-eval-section">
      <div class="summary-eval-title">
        Quality Check ·
        <span style="color:${overallColor};font-style:normal;">${escHtml(overall)}</span>
      </div>
      ${warningHtml}
      <div class="summary-eval-grid">${cardsHtml}</div>
      ${thoughtHtml}
    </div>`;
}

function renderSummaryCompetition(competition) {
  if (!competition || !Array.isArray(competition.scorecards) || competition.scorecards.length === 0) {
    return '';
  }

  const repair = (competition.repair_status || 'not_needed').replace(/_/g, ' ');
  const cardsHtml = competition.scorecards.map(card => {
    const selected = card.selected ? ' selected' : '';
    const faithClass = card.faithfulness === 'pass' ? 'pass' : card.faithfulness === 'fail' ? 'fail' : 'na';
    const score = typeof card.score === 'number' ? card.score.toFixed(2) : '—';
    return `
      <div class="summary-competition-card${selected}">
        <div class="summary-competition-head">
          <span>${escHtml(card.rank ? `#${card.rank} · ${card.name}` : card.name || 'Candidate')}</span>
          <strong>${escHtml(score)}</strong>
        </div>
        <div class="summary-competition-meta">
          <span class="summary-competition-faith ${faithClass}">faithfulness ${escHtml(card.faithfulness || 'n/a')}</span>
          ${card.selected ? '<span class="summary-competition-winner">winner</span>' : ''}
        </div>
      </div>`;
  }).join('');

  return `
    <div class="summary-competition-section">
      <div class="summary-competition-title">Competitive Summary Selection</div>
      <div class="summary-competition-winner-line">
        Selected <strong>${escHtml(competition.winner_name || 'best candidate')}</strong>
        ${typeof competition.winner_score === 'number' ? `with score ${escHtml(competition.winner_score.toFixed(2))}` : ''}.
      </div>
      <div class="summary-competition-rationale">${escHtml(competition.selection_rationale || '')}</div>
      <div class="summary-competition-repair">Repair status: ${escHtml(repair)}</div>
      <div class="summary-competition-grid">${cardsHtml}</div>
    </div>`;
}

// ── Approval modal ─────────────────────────────────────────────────────────────
function showApprovalModal(interruptPayload) {
  const modal = document.getElementById('approval-modal');
  const title = document.getElementById('modal-title');
  const body = document.getElementById('modal-body');
  modal.classList.remove('hidden');

  if (interruptPayload.pause_point === 'before_download') {
    const paper = interruptPayload.paper || {};
    title.textContent = 'Analyze this paper in depth?';
    body.innerHTML = `
      <p style="margin-bottom:10px;"><strong>${escHtml(paper.title || 'Selected paper')}</strong></p>
      <p style="margin-bottom:8px;">${escHtml(interruptPayload.message || '')}</p>
      <p style="color:var(--amber);">⏱ Estimated time: ${interruptPayload.estimated_time || '30-60s'}</p>
      <p style="color:var(--text-muted);margin-top:6px;font-size:12px;">${escHtml(interruptPayload.warning || '')}</p>
    `;
    document.getElementById('btn-revise').classList.add('hidden');
    document.getElementById('btn-approve').textContent = 'Yes, analyze';
  } else if (interruptPayload.pause_point === 'before_save') {
    title.textContent = 'Save this summary to your library?';
    const s = interruptPayload.draft_summary || {};
    const evalHtml = renderSummaryEvaluation(interruptPayload.summary_evaluation);
    const competitionHtml = renderSummaryCompetition(interruptPayload.summary_competition);
    body.innerHTML = `
      <p style="margin-bottom:10px;">${escHtml(interruptPayload.message || '')}</p>
      ${s.overview ? `<div class="summary-preview-box">${escHtml(s.overview)}</div>` : ''}
      ${competitionHtml}
      ${evalHtml}
    `;
    document.getElementById('btn-revise').classList.remove('hidden');
    document.getElementById('btn-approve').textContent = 'Approve & Save';
  }
}

function hideModal() {
  document.getElementById('approval-modal').classList.add('hidden');
  document.getElementById('revision-input').classList.add('hidden');
  document.getElementById('btn-submit-revision').classList.add('hidden');
  document.getElementById('btn-revise').classList.remove('hidden');
}

async function submitReview(decision, revisionNote = '') {
  hideModal();
  const session_id = state.currentSession;
  if (!session_id) return;

  try {
    await apiPost('/api/review/decide', {
      session_id,
      decision,
      revision_note: revisionNote,
    });

    if (decision === 'approved') {
      updateStatusBar('Processing...', 'running');
      startPolling(session_id);
    } else if (decision === 'rejected') {
      updateStatusBar('Cancelled', 'idle');
      document.getElementById('tab-summary').innerHTML =
        '<div style="padding:16px;color:var(--text-muted);">Analysis cancelled.</div>';
    } else if (decision === 'revised') {
      updateStatusBar('Regenerating summary...', 'running');
      startPolling(session_id);
    }
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Q/A side-panel tab switching ───────────────────────────────────────────────
function switchQaSideTab(name) {
  document.querySelectorAll('.qa-side-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.qaside === name);
  });
  document.getElementById('qa-side-tools').classList.toggle('hidden', name !== 'tools');
  document.getElementById('qa-side-logs').classList.toggle('hidden', name !== 'logs');
  document.getElementById('qa-side-tracking').classList.toggle('hidden', name !== 'tracking');
}

// ── Event log ─────────────────────────────────────────────────────────────────
function appendLog(level, message) {
  const container = document.getElementById('qa-log-entries');
  if (!container) return;
  // Clear placeholder text on first real entry
  if (container.classList.contains('qa-empty')) {
    container.innerHTML = '';
    container.classList.remove('qa-empty');
  }
  const time = new Date().toLocaleTimeString('en-US', { hour12: false });
  const entry = document.createElement('div');
  entry.className = `qa-log-entry qa-log-${level}`;
  entry.innerHTML =
    `<span class="qa-log-time">${time}</span>` +
    `<span class="qa-log-msg">${escHtml(message)}</span>`;
  container.appendChild(entry);
  container.scrollTop = container.scrollHeight;
  state.qaLogCount++;
  // Badge the Logs tab when it isn't the active pane
  const logsTab = document.querySelector('.qa-side-tab[data-qaside="logs"]');
  if (logsTab && !logsTab.classList.contains('active')) {
    logsTab.textContent = `Logs (${state.qaLogCount})`;
  }
}

function clearQaLog() {
  state.qaLogCount = 0;
  const container = document.getElementById('qa-log-entries');
  if (!container) return;
  container.innerHTML = '<span class="qa-log-placeholder">No activity yet. Ask a question to see logs.</span>';
  container.classList.add('qa-empty');
  const logsTab = document.querySelector('.qa-side-tab[data-qaside="logs"]');
  if (logsTab) logsTab.textContent = 'Logs';
}

// ── Q&A flow ───────────────────────────────────────────────────────────────────
async function sendQuestion(question) {
  if (!state.currentPaper || !question.trim()) return;

  resetPdfViewer();
  clearQaLog();
  appendLog('info', `Q/A started — "${question}"`);
  renderChatMessage('user', question);
  const typingEl = addTypingIndicator();

  if (state.qaThreadPaperId !== state.currentPaper.arxiv_id) {
    state.qaThreadSessionId = null;
    state.qaThreadPaperId = state.currentPaper.arxiv_id;
  }
  const session_id = state.qaThreadSessionId || generateId();
  state.qaThreadSessionId = session_id;
  state.currentSession = session_id;
  state.currentQaResult = null;
  updateStatusBar('Running MCP Q/A...', 'running');

  try {
    appendLog('info', 'Sending request to Q/A orchestrator...');
    const result = await apiPost('/api/qa', {
      message: question,
      arxiv_id: state.currentPaper.arxiv_id,
      session_id,
    });
    appendLog('info', `Session ${result.session_id?.slice(0, 8) ?? '?'} — MCP pipeline running`);

    // Poll for answer
    const pollForAnswer = async () => {
      let seenTimeline = 0;
      for (let i = 0; i < 120; i++) {
        await sleep(1500);
        const status = await apiGet(`/api/qa/status/${result.session_id}`);
        const timeline = status.tool_timeline || [];
        renderToolTimeline(timeline);
        renderTracking(status.tracking);
        // Log only new timeline entries since last poll
        const newEntries = timeline.slice(seenTimeline);
        newEntries.forEach(entry => {
          const lvl = entry.kind === 'rationale' ? 'info'
                    : entry.status === 'failed' ? 'err'
                    : entry.status === 'completed' ? 'ok'
                    : 'info';
          const detail = entry.details ? ` — ${entry.details}` : '';
          appendLog(lvl, `${entry.kind === 'rationale' ? 'CoT Trace: ' : ''}${entry.title}${detail}`);
        });
        seenTimeline = timeline.length;
        if (status.status === 'completed') {
          typingEl.remove();
          state.currentQaResult = status;
          if (status.chat_message) {
            renderChatMessage('assistant', status.chat_message, status.answer_citations || []);
          }
          renderQaArtifacts(status);
          appendLog('ok', 'Q/A complete');
          updateStatusBar('Q/A complete', 'ok');
          return;
        }
        if (status.status === 'error') {
          typingEl.remove();
          renderChatMessage('assistant', `Error: ${status.error}`, []);
          appendLog('err', `Q/A failed — ${status.error || 'unknown error'}`);
          updateStatusBar('Q/A failed', 'error');
          return;
        }
      }
      typingEl.remove();
      renderChatMessage('assistant', 'Request timed out. Please try again.', []);
      appendLog('warn', 'Q/A timed out after 180 s');
      updateStatusBar('Q/A timed out', 'error');
    };

    pollForAnswer();
  } catch (err) {
    typingEl.remove();
    renderChatMessage('assistant', `Error: ${err.message}`, []);
    appendLog('err', `Request failed — ${err.message}`);
    updateStatusBar('Q/A failed', 'error');
  }
}

async function loadQaTools(force = false) {
  if (!force && state.availableQaTools.length > 0) {
    renderAvailableTools(state.availableQaTools);
    return;
  }
  try {
    const result = await apiGet('/api/qa/tools');
    state.availableQaTools = result.tools || [];
    renderAvailableTools(state.availableQaTools);
    appendLog('ok', `MCP server ready — ${state.availableQaTools.length} tools available`);
  } catch (err) {
    const container = document.getElementById('qa-available-tools');
    container.innerHTML = 'Could not load the MCP tool list.';
    container.classList.add('qa-empty');
    appendLog('err', `MCP tool list failed to load — ${err.message}`);
  }
}

async function clearQaAssets() {
  const hasActiveAssets = state.currentQaResult?.assets?.length || 0;
  const confirmed = window.confirm(
    hasActiveAssets
      ? 'Delete all generated Q/A files from disk? This will remove current downloads and graphics.'
      : 'Delete all generated Q/A files from disk?'
  );
  if (!confirmed) return;

  try {
    const result = await apiDelete('/api/qa/assets');
    showToast(`Cleared ${result.removed_sessions} Q/A asset folder${result.removed_sessions !== 1 ? 's' : ''}.`, 'success');
    document.getElementById('qa-assets').innerHTML = 'Generated downloads will appear here.';
    document.getElementById('qa-assets').classList.add('qa-empty');
    document.getElementById('qa-graphic').innerHTML = '';
    document.getElementById('qa-graphic-panel').classList.add('hidden');
    if (state.currentQaResult) {
      state.currentQaResult.assets = [];
      state.currentQaResult.generated_image = null;
    }
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function renderChatMessage(role, content, citations = []) {
  const messages = document.getElementById('chat-messages');
  const el = document.createElement('div');
  el.className = `chat-message ${role}`;
  el.innerHTML = `
    <div class="chat-message-body">${escHtml(content)}</div>
    ${citations.length > 0 ? `
      <div class="chat-citations">
        ${citations.map((c, idx) => `
          <span class="citation-badge" data-evidence-index="${idx}">
            [${escHtml(c.section || 'Section')}, p.${escHtml(c.page || '?')}]
          </span>
        `).join('')}
      </div>
    ` : ''}
  `;
  messages.appendChild(el);
  el.querySelectorAll('[data-evidence-index]').forEach(btn => {
    btn.addEventListener('click', () => openEvidenceItem(parseInt(btn.dataset.evidenceIndex, 10)));
  });
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function addTypingIndicator() {
  const messages = document.getElementById('chat-messages');
  const el = document.createElement('div');
  el.className = 'typing-indicator';
  el.innerHTML = `<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>`;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

function resetQaWorkspace() {
  document.getElementById('chat-messages').innerHTML = '';
  document.getElementById('qa-available-tools').innerHTML = 'Loading MCP tool list...';
  document.getElementById('qa-available-tools').classList.add('qa-empty');
  document.getElementById('qa-tool-timeline').innerHTML = 'Ask a question to see MCP tool usage.';
  document.getElementById('qa-tool-timeline').classList.add('qa-empty');
  document.getElementById('qa-tracking-status').innerHTML = 'Ask a question to run the LLM-as-judge evaluation.';
  document.getElementById('qa-tracking-status').classList.add('qa-empty');
  document.getElementById('qa-assets').innerHTML = 'Generated downloads will appear here.';
  document.getElementById('qa-assets').classList.add('qa-empty');
  document.getElementById('qa-evidence-list').innerHTML = 'Evidence quotes and PDF highlights will appear here.';
  document.getElementById('qa-evidence-list').classList.add('qa-empty');
  document.getElementById('qa-graphic').innerHTML = '';
  document.getElementById('qa-graphic-panel').classList.add('hidden');
  document.getElementById('qa-evidence-panel').classList.add('hidden');
  state.currentQaResult = null;
  state.qaThreadSessionId = null;
  state.qaThreadPaperId = null;
  resetPdfViewer();
  loadQaTools();
}

function renderToolTimeline(timeline) {
  const container = document.getElementById('qa-tool-timeline');
  if (!timeline || timeline.length === 0) {
    container.innerHTML = 'Ask a question to see MCP tool usage.';
    container.classList.add('qa-empty');
    return;
  }
  container.classList.remove('qa-empty');
  container.innerHTML = timeline.map(step => {
    if (step.kind === 'rationale') {
      return `
        <div class="qa-cot-entry">
          <div class="qa-cot-label">${escHtml(step.title || 'CoT Trace')}</div>
          <div class="qa-cot-text">${escHtml(step.details || '')}</div>
        </div>
      `;
    }
    return `
      <div class="qa-tool-step">
        <div class="qa-tool-step-title">${escHtml(step.title || step.tool || 'Tool')}</div>
        <div class="qa-tool-step-meta">${escHtml(step.details || '')}</div>
      </div>
    `;
  }).join('');
}

function renderQaArtifacts(result) {
  renderAvailableTools(result.available_tools || []);
  renderToolTimeline(result.tool_timeline || []);
  renderTracking(result.tracking);
  renderQaAssets(result.assets || []);
  renderGeneratedGraphic(result.generated_image);
  renderEvidenceBundle(result.evidence_bundle || { items: [] });
}

function metricClass(status) {
  if (status === 'pass') return 'pass';
  if (status === 'fail') return 'fail';
  return 'na';
}

function prettyMetricName(name) {
  return String(name || '')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, ch => ch.toUpperCase());
}

function renderTracking(tracking) {
  const container = document.getElementById('qa-tracking-status');
  if (!container) return;
  if (!tracking) {
    container.innerHTML = 'Evaluation will appear after the judge runs.';
    container.classList.add('qa-empty');
    return;
  }

  // Ordered to match METRICS tuple in evaluation.py
  const metrics = [
    'answer_relevance',
    'groundedness',
    'citation_quality',
    'retrieval_relevance',
    'tool_choice_quality',
    'artifact_match',
  ];
  const overall = tracking.overall_status || 'needs_review';
  const phoenix = tracking.phoenix || {};
  const toolCounts = tracking.tool_counts || {};
  const toolCountHtml = Object.keys(toolCounts).length
    ? Object.entries(toolCounts).map(([name, count]) => `<span class="qa-tool-count">${escHtml(name)} × ${escHtml(count)}</span>`).join('')
    : '<span class="qa-empty">No completed tool calls counted yet.</span>';

  // Build CoT thought sections if the judge returned them
  const qualityThought = tracking.quality_thought || '';
  const supportingThought = tracking.supporting_thought || '';
  const thoughtHtml = (qualityThought || supportingThought) ? `
    <div class="qa-panel-title qa-tracking-subtitle">Judge Reasoning</div>
    ${qualityThought ? `<div class="qa-cot-block"><span class="qa-cot-label-inline">Quality:</span> ${escHtml(qualityThought)}</div>` : ''}
    ${supportingThought ? `<div class="qa-cot-block"><span class="qa-cot-label-inline">Supporting:</span> ${escHtml(supportingThought)}</div>` : ''}
  ` : '';
  const calibrationNotes = Array.isArray(tracking.calibration_notes) ? tracking.calibration_notes : [];
  const calibrationHtml = calibrationNotes.length ? `
    <div class="qa-repair-note">
      <strong>Evaluator calibration:</strong>
      ${calibrationNotes.map(note => `<div>${escHtml(note)}</div>`).join('')}
    </div>
  ` : '';

  // Phoenix dashboard link — shown only when Phoenix is enabled
  const phoenixEnabled = phoenix.status === 'ready';
  const phoenixLinkHtml = phoenixEnabled
    ? `<a href="http://localhost:6006" target="_blank" rel="noopener noreferrer" class="qa-phoenix-link">View traces in Phoenix ↗</a>`
    : '';

  container.classList.remove('qa-empty');
  container.innerHTML = `
    <div class="qa-tracking-summary qa-tracking-${escHtml(overall)}">
      <div class="qa-tracking-kicker">Overall</div>
      <div class="qa-tracking-title">${escHtml(overall.replaceAll('_', ' '))}</div>
      <div class="qa-tracking-meta">
        Source: ${escHtml(tracking.source || 'judge')} · Phoenix: ${escHtml(phoenix.status || 'unknown')}
      </div>
    </div>
    <div class="qa-tracking-grid">
      ${metrics.map(name => {
        const item = tracking[name] || {};
        const status = item.status || 'not_applicable';
        const score = typeof item.score === 'number' ? item.score.toFixed(2) : '—';
        return `
          <div class="qa-score-card qa-score-${metricClass(status)}">
            <div class="qa-score-head">
              <span>${escHtml(prettyMetricName(name))}</span>
              <strong>${escHtml(score)}</strong>
            </div>
            <div class="qa-score-status">${escHtml(status.replaceAll('_', ' '))}</div>
            <div class="qa-score-note">${escHtml(item.note || '')}</div>
          </div>
        `;
      }).join('')}
    </div>
    ${tracking.repair_reason ? `<div class="qa-repair-note"><strong>Repair note:</strong> ${escHtml(tracking.repair_reason)}</div>` : ''}
    ${calibrationHtml}
    ${tracking.evaluation_error ? `<div class="qa-repair-note qa-score-fail"><strong>Eval error:</strong> ${escHtml(tracking.evaluation_error)}</div>` : ''}
    ${thoughtHtml}
    <div class="qa-panel-title qa-tracking-subtitle">Tool Counts</div>
    <div class="qa-tool-counts">${toolCountHtml}</div>
    <div class="qa-panel-title qa-tracking-subtitle">Phoenix Trace</div>
    <div class="qa-score-note">${escHtml(phoenix.message || 'Phoenix status unavailable.')}</div>
    ${phoenixLinkHtml}
  `;
}

function renderAvailableTools(tools) {
  const container = document.getElementById('qa-available-tools');
  if (!tools || tools.length === 0) {
    container.innerHTML = 'Ask a question to load the MCP tool list.';
    container.classList.add('qa-empty');
    return;
  }
  container.classList.remove('qa-empty');
  container.innerHTML = tools.map(tool => `
    <div class="qa-tool-card">
      <div class="qa-tool-step-title">${escHtml(tool.name || 'tool')}</div>
      <div class="qa-tool-step-meta">${escHtml(tool.description || '')}</div>
    </div>
  `).join('');
}

function renderQaAssets(assets) {
  const container = document.getElementById('qa-assets');
  const downloads = (assets || []).filter(a => ['markdown', 'pdf', 'presentation'].includes(a.kind));
  if (downloads.length === 0) {
    container.innerHTML = 'Generated downloads will appear here.';
    container.classList.add('qa-empty');
    return;
  }
  container.classList.remove('qa-empty');
  container.innerHTML = downloads.map(asset => `
    <div class="qa-asset-card">
      <div>
        <div class="qa-tool-step-title">${escHtml(asset.label || asset.filename)}</div>
        <div class="qa-tool-step-meta">${escHtml(asset.filename || '')}</div>
      </div>
      <a class="btn-ghost" href="${asset.url}" download>Download</a>
    </div>
  `).join('');
}

function renderGeneratedGraphic(asset) {
  const panel = document.getElementById('qa-graphic-panel');
  const container = document.getElementById('qa-graphic');
  if (!asset || !asset.url) {
    panel.classList.add('hidden');
    container.innerHTML = '';
    return;
  }
  panel.classList.remove('hidden');
  container.innerHTML = `<img src="${asset.url}" alt="${escHtml(asset.label || 'Generated graphic')}" />`;
  // "View full size" link sits outside the scrollable image box
  let link = document.getElementById('qa-graphic-link');
  if (!link) {
    link = document.createElement('a');
    link.id = 'qa-graphic-link';
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = 'View full size ↗';
    panel.appendChild(link);
  }
  link.href = asset.url;
}

function renderEvidenceBundle(bundle) {
  const panel = document.getElementById('qa-evidence-panel');
  const container = document.getElementById('qa-evidence-list');
  const items = bundle?.items || [];
  if (!bundle?.enabled || items.length === 0) {
    panel.classList.add('hidden');
    container.innerHTML = 'Evidence quotes will appear here.';
    container.classList.add('qa-empty');
    resetPdfViewer();
    return;
  }
  panel.classList.remove('hidden');
  container.classList.remove('qa-empty');
  container.innerHTML = items.map((item, idx) => `
    <div class="qa-evidence-item" data-evidence-index="${idx}">
      <div class="qa-evidence-meta">${escHtml(item.section || 'Unknown')} · page ${escHtml(item.page || '?')}</div>
      <div class="qa-evidence-quote">${escHtml(item.quote || '')}</div>
    </div>
  `).join('');
  container.querySelectorAll('[data-evidence-index]').forEach(el => {
    el.addEventListener('click', () => openEvidenceItem(parseInt(el.dataset.evidenceIndex, 10)));
  });
  openEvidenceItem(0);
}

function resetPdfViewer() {
  state.pdfDoc = null;
  state.pdfUrl = null;
  state.pdfCurrentPage = 1;
  state.pdfScale = 1.2;
  state.pdfEvidenceIndex = null;
  const canvas = document.getElementById('qa-pdf-canvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width || 1, canvas.height || 1);
  }
  document.getElementById('qa-page-indicator').textContent = 'Page 0 / 0';
}

async function ensurePdfDocument(url) {
  if (!window.pdfjsLib) throw new Error('PDF viewer dependency is unavailable.');
  if (state.pdfDoc && state.pdfUrl === url) return state.pdfDoc;
  state.pdfUrl = url;
  state.pdfDoc = await window.pdfjsLib.getDocument(url).promise;
  return state.pdfDoc;
}

async function renderPdfPage(pageNumber) {
  if (!state.pdfDoc) return;
  const page = await state.pdfDoc.getPage(pageNumber);
  const viewport = page.getViewport({ scale: state.pdfScale });
  const canvas = document.getElementById('qa-pdf-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  await page.render({ canvasContext: ctx, viewport }).promise;
  document.getElementById('qa-page-indicator').textContent = `Page ${pageNumber} / ${state.pdfDoc.numPages}`;
}

async function openEvidenceItem(index) {
  const bundle = state.currentQaResult?.evidence_bundle;
  const items = bundle?.items || [];
  if (!items[index]) return;

  state.pdfEvidenceIndex = index;
  document.querySelectorAll('.qa-evidence-item, .citation-badge').forEach(el => el.classList.remove('active'));
  document.querySelectorAll(`[data-evidence-index="${index}"]`).forEach(el => el.classList.add('active'));

  const item = items[index];
  const pdfUrl = bundle?.pdf_url;
  if (!pdfUrl) return;
  try {
    const doc = await ensurePdfDocument(pdfUrl);
    // Page numbers from chunker are estimated from char position in cleaned text,
    // which strips preamble pages — actual PDF page is 1 ahead
    const pageNumber = Math.max(1, Math.min((item.page || 1) + 1, doc.numPages));
    state.pdfCurrentPage = pageNumber;
    await renderPdfPage(pageNumber);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Library ────────────────────────────────────────────────────────────────────
async function loadLibrary() {
  const pane = document.getElementById('tab-library');
  pane.innerHTML = '<div style="color:var(--text-muted);font-size:13px;">Loading library...</div>';

  try {
    const papers = await apiGet('/api/library');
    state.library = papers || [];
    state.libraryIds = new Set(state.library.map(p => p.arxiv_id));

    if (!papers || papers.length === 0) {
      pane.innerHTML = '<div style="color:var(--text-muted);font-size:13px;">Your library is empty. Analyze and approve papers to add them here.</div>';
      return;
    }

    pane.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <span style="font-size:13px;color:var(--text-muted)">${papers.length} saved paper${papers.length !== 1 ? 's' : ''}</span>
        <button class="btn-ghost" id="btn-clear-library" style="font-size:12px;padding:4px 12px;color:var(--red);border-color:rgba(248,113,113,0.3);">
          Delete All
        </button>
      </div>
      ${papers.map(p => `
        <div class="library-card" data-arxiv="${p.arxiv_id}" role="button" tabindex="0" title="Open saved paper">
          <div class="library-card-title">${escHtml(p.title || p.arxiv_id)}</div>
          <div class="library-card-date">
            <span style="font-family:var(--font-mono)">${p.arxiv_id}</span>
            · Saved ${formatDate(p.saved_at)}
          </div>
          <div class="library-card-actions">
            <div class="star-rating" data-arxiv="${p.arxiv_id}" data-current="${p.rating || 0}">
              ${[1,2,3,4,5].map(n => `<span class="star ${(p.rating || 0) >= n ? 'filled' : ''}" data-rating="${n}">★</span>`).join('')}
            </div>
            <button class="btn-ghost" data-remove="${p.arxiv_id}" style="font-size:12px;padding:4px 10px;">Remove</button>
          </div>
        </div>
      `).join('')}
    `;

    // "Delete All" button
    document.getElementById('btn-clear-library').addEventListener('click', clearLibrary);

    // Click a saved paper card to reopen it with its summary and cached metadata.
    pane.querySelectorAll('.library-card').forEach(card => {
      const openCard = () => {
        const saved = state.library.find(p => p.arxiv_id === card.dataset.arxiv);
        if (saved) openSavedPaper(saved);
      };
      card.addEventListener('click', openCard);
      card.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          openCard();
        }
      });
    });

    // Per-card remove buttons (event delegation avoids onclick in HTML)
    pane.querySelectorAll('[data-remove]').forEach(btn => {
      btn.addEventListener('click', (event) => {
        event.stopPropagation();
        removeFromLibrary(btn.dataset.remove);
      });
    });

    // Star rating click handlers — persist immediately to DB
    pane.querySelectorAll('.star-rating').forEach(ratingEl => {
      const arxivId = ratingEl.dataset.arxiv;
      ratingEl.addEventListener('click', event => event.stopPropagation());
      ratingEl.querySelectorAll('.star').forEach(star => {
        star.addEventListener('click', async (event) => {
          event.stopPropagation();
          const rating = parseInt(star.dataset.rating);
          // Update visuals instantly
          ratingEl.querySelectorAll('.star').forEach((s, i) => {
            s.classList.toggle('filled', i < rating);
          });
          ratingEl.dataset.current = rating;
          await submitFeedback(arxivId, rating);
        });
      });
    });
  } catch (err) {
    pane.innerHTML = `<div style="color:var(--red);font-size:13px;">${err.message}</div>`;
  }
}

async function clearLibrary() {
  if (!confirm('Delete all papers from your library? This cannot be undone.')) return;
  try {
    const result = await apiDelete('/api/library');
    showToast(`Deleted ${result.count} paper${result.count !== 1 ? 's' : ''}`, 'success');
    loadLibrary();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function removeFromLibrary(arxiv_id) {
  try {
    await apiDelete(`/api/library/${encodeURIComponent(arxiv_id)}`);
    // Remove card from DOM without full reload
    const card = document.querySelector(`.library-card[data-arxiv="${arxiv_id}"]`);
    if (card) card.remove();
    showToast('Paper removed', 'success');
    // Reload to update count header
    loadLibrary();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function submitFeedback(arxiv_id, rating, comment = '') {
  try {
    // /feedback saves to both saved_papers.rating AND paper_feedback + updates prefs
    await apiPost(`/api/library/${encodeURIComponent(arxiv_id)}/feedback`, { rating, comment });
    showToast('Rating saved — preferences updated', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Preferences panel ──────────────────────────────────────────────────────────
async function loadPreferences() {
  const pane = document.getElementById('tab-preferences');
  pane.innerHTML = '<div style="color:var(--text-muted);font-size:13px;">Loading preferences...</div>';

  try {
    const prefs = await apiGet('/api/preferences');
    if (!prefs || prefs.length === 0) {
      pane.innerHTML = '<div style="color:var(--text-muted);font-size:13px;">No preferences yet. Rate papers to build your preference profile.</div>';
      return;
    }

    pane.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <span style="font-size:13px;color:var(--text-muted);">Topic interest weights — updated by ratings and saves.</span>
        <button class="btn-ghost" id="btn-clear-prefs" style="font-size:12px;padding:4px 12px;color:var(--red);border-color:rgba(248,113,113,0.3);">
          Reset All
        </button>
      </div>
      ${prefs.map(p => `
        <div class="pref-bar-row">
          <span class="pref-label">${escHtml(p.topic)}</span>
          <div class="pref-bar-wrap">
            <div class="pref-bar" style="width:${Math.round(p.weight * 100)}%"></div>
          </div>
          <span class="pref-weight">${Math.round(p.weight * 100)}%</span>
        </div>
      `).join('')}
    `;

    document.getElementById('btn-clear-prefs').addEventListener('click', clearPreferences);
  } catch (err) {
    pane.innerHTML = `<div style="color:var(--red);font-size:13px;">${err.message}</div>`;
  }
}

async function clearPreferences() {
  if (!confirm('Reset all topic preferences? This cannot be undone.')) return;
  try {
    await apiDelete('/api/preferences');
    showToast('Preferences reset', 'success');
    loadPreferences();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ── Shortlist ──────────────────────────────────────────────────────────────────
async function shortlistPaper(arxiv_id, btn) {
  btn.classList.toggle('starred');
  btn.textContent = btn.classList.contains('starred') ? '★' : '☆';

  if (btn.classList.contains('starred')) {
    if (!state.shortlist.includes(arxiv_id)) {
      state.shortlist.push(arxiv_id);
    }
    showToast('Added to shortlist', 'success');
    renderShortlist();
  }
}

function renderShortlist() {
  const section = document.getElementById('shortlist-section');
  const items = document.getElementById('shortlist-items');
  if (state.shortlist.length === 0) { section.classList.add('hidden'); return; }
  section.classList.remove('hidden');
  items.innerHTML = state.shortlist.map(id => `
    <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted);padding:4px 0;">${id}</div>
  `).join('');
}

// ── Co-author network ──────────────────────────────────────────────────────────
function renderCoauthorNetwork(graph) {
  const container = document.getElementById('tab-panel-network');
  if (!container) return;
  container.innerHTML = '';

  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    container.innerHTML = '<p style="padding:20px;color:var(--ink-2)">No co-author data available.</p>';
    return;
  }

  // Ambiguity warning
  if (graph.ambiguity_warning) {
    const warn = document.createElement('div');
    warn.className = 'coauthor-warning';
    warn.textContent = '\u26A0 Results may include multiple authors with this name. Disconnected clusters in the graph indicate possible name ambiguity.';
    container.appendChild(warn);
  }

  // D3 fallback: if D3 not available, render simple text list
  if (typeof d3 === 'undefined') {
    const fallback = document.createElement('div');
    fallback.style.cssText = 'padding:20px;';
    const queryNode = graph.nodes.find(n => n.is_query_author);
    const others = graph.nodes.filter(n => !n.is_query_author)
      .sort((a, b) => b.paper_count - a.paper_count).slice(0, 20);
    fallback.innerHTML = `
      <p style="color:var(--ink-2);margin-bottom:12px;">
        <strong>${escHtml(queryNode ? queryNode.label : graph.query_author)}</strong>
        — ${graph.nodes.length} authors, ${graph.edges.length} collaborations
      </p>
      <p style="color:var(--ink-3);font-size:12px;margin-bottom:8px;">Top collaborators:</p>
      <ul style="list-style:none;padding:0;">
        ${others.map(n => `<li style="padding:4px 0;font-size:13px;color:var(--ink-2);">
          ${escHtml(n.label)} <span style="color:var(--ink-3);font-size:11px;">(${n.paper_count} paper${n.paper_count !== 1 ? 's' : ''})</span>
        </li>`).join('')}
      </ul>
    `;
    container.appendChild(fallback);
    return;
  }

  // Category color scale
  const categories = [...new Set(graph.nodes.map(n => n.primary_category))];
  const colorScale = d3.scaleOrdinal(d3.schemeTableau10).domain(categories);

  const width = container.clientWidth || 900;
  const height = Math.max(560, Math.round(width * 0.65));

  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .style('background', 'var(--paper)');

  // Tooltip div
  const tooltip = d3.select(container)
    .append('div')
    .attr('class', 'coauthor-tooltip')
    .style('opacity', 0);

  const links = graph.edges.map(e => ({
    source: e.source,
    target: e.target,
    weight: e.weight,
    titles: e.titles,
  }));

  const maxPapers = d3.max(graph.nodes, n => n.paper_count) || 1;
  const nodeRadius = n => n.is_query_author ? 36 : Math.max(12, 10 + (n.paper_count / maxPapers) * 22);
  const edgeWidth = e => Math.max(2, Math.min(10, e.weight * 2.5));

  // Deep-copy nodes so D3 can mutate x/y
  const simNodes = graph.nodes.map(n => ({...n}));

  const simulation = d3.forceSimulation(simNodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(140))
    .force('charge', d3.forceManyBody().strength(-400))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 8));

  state._d3Simulation = simulation;

  // Zoom/pan — all graph elements live inside a single <g> that gets transformed
  const zoomLayer = svg.append('g').attr('class', 'zoom-layer');
  const zoom = d3.zoom()
    .scaleExtent([0.2, 4])
    .on('zoom', (event) => zoomLayer.attr('transform', event.transform));
  svg.call(zoom);

  // Hint text that fades after first interaction
  svg.append('text')
    .attr('x', 10).attr('y', 18)
    .attr('font-size', 10).attr('fill', 'var(--ink-3)')
    .attr('class', 'zoom-hint')
    .text('Scroll to zoom · Drag to pan · Click node to filter');

  const link = zoomLayer.append('g')
    .selectAll('line')
    .data(links)
    .join('line')
    .attr('stroke', '#ccc')
    .attr('stroke-width', d => edgeWidth(d))
    .attr('stroke-opacity', 0.7)
    .on('mouseover', (event, d) => {
      tooltip.transition().duration(150).style('opacity', 0.97);
      const titles = d.titles && d.titles.length > 0 ? d.titles.map(t => `\u2022 ${t}`).join('<br>') : 'Shared paper';
      tooltip.html(`<strong>${d.weight} shared paper${d.weight > 1 ? 's' : ''}</strong><br>${titles}`)
        .style('left', (event.offsetX + 12) + 'px')
        .style('top', (event.offsetY - 10) + 'px');
    })
    .on('mouseout', () => tooltip.transition().duration(200).style('opacity', 0));

  const node = zoomLayer.append('g')
    .selectAll('circle')
    .data(simNodes)
    .join('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => d.is_query_author ? '#f5c518' : colorScale(d.primary_category))
    .attr('stroke', d => d.is_query_author ? '#c8960c' : 'white')
    .attr('stroke-width', d => d.is_query_author ? 3 : 1.5)
    .style('cursor', 'pointer')
    .call(d3.drag()
      .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    )
    .on('mouseover', (event, d) => {
      tooltip.transition().duration(150).style('opacity', 0.97);
      tooltip.html(`<strong>${escHtml(d.label)}</strong><br>${d.paper_count} paper${d.paper_count !== 1 ? 's' : ''}<br><em>${escHtml(d.primary_category)}</em>`)
        .style('left', (event.offsetX + 12) + 'px')
        .style('top', (event.offsetY - 10) + 'px');
    })
    .on('mouseout', () => tooltip.transition().duration(200).style('opacity', 0))
    .on('click', (event, d) => {
      filterByAuthor(d.id, d.label, d.paper_ids);
    });

  const label = zoomLayer.append('g')
    .selectAll('text')
    .data(simNodes)
    .join('text')
    .text(d => d.label.split(' ').slice(-1)[0])
    .attr('font-size', d => d.is_query_author ? 12 : 10)
    .attr('font-weight', d => d.is_query_author ? '700' : '400')
    .attr('fill', 'var(--ink)')
    .attr('text-anchor', 'middle')
    .attr('dy', d => nodeRadius(d) + 13)
    .style('pointer-events', 'none');

  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    label.attr('x', d => d.x).attr('y', d => d.y);
  });

  // Legend
  const legend = document.createElement('div');
  legend.className = 'coauthor-legend';
  legend.innerHTML = categories.slice(0, 6).map(cat =>
    `<span class="legend-dot" style="background:${colorScale(cat)}"></span>${escHtml(cat)}`
  ).join('  ');
  container.appendChild(legend);
}

function filterByAuthor(authorId, authorLabel, paperIds) {
  state.activeAuthorFilter = authorId;
  const paperIdSet = new Set(paperIds);

  // Show/hide cards based on paper_ids
  document.querySelectorAll('.paper-card').forEach(card => {
    const id = card.dataset.arxivId;
    card.style.display = paperIdSet.has(id) ? '' : 'none';
  });

  // Show clear-filter chip
  let chip = document.getElementById('author-filter-chip');
  if (!chip) {
    chip = document.createElement('div');
    chip.id = 'author-filter-chip';
    chip.className = 'author-filter-chip';
    const resultsList = document.getElementById('results-list');
    resultsList.parentNode.insertBefore(chip, resultsList);
  }
  chip.innerHTML = `Showing papers by <strong>${escHtml(authorLabel)}</strong> <button onclick="clearAuthorFilter()">\u2715 Clear</button>`;
  chip.classList.remove('hidden');
}

function clearAuthorFilter() {
  state.activeAuthorFilter = null;
  document.querySelectorAll('.paper-card').forEach(card => card.style.display = '');
  const chip = document.getElementById('author-filter-chip');
  if (chip) chip.classList.add('hidden');
}

// ── UI helpers ─────────────────────────────────────────────────────────────────
function updateStatusBar(text, type = 'idle') {
  const indicator = document.getElementById('status-indicator');
  const textEl = document.getElementById('status-text');
  textEl.textContent = text;
  indicator.className = `dot dot-${type === 'running' ? 'running' : type === 'ok' ? 'ok' : type === 'error' ? 'error' : 'idle'}`;
}

function showToast(message, type = 'success') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

function setLoading(isLoading) {
  const btn = document.getElementById('search-btn');
  const text = document.getElementById('search-btn-text');
  const spinner = document.getElementById('search-spinner');
  btn.disabled = isLoading;
  text.classList.toggle('hidden', isLoading);
  spinner.classList.toggle('hidden', !isLoading);
}

function switchTab(tabName) {
  // Stop D3 simulation when switching away from network tab
  if (tabName !== 'network' && state._d3Simulation) {
    state._d3Simulation.stop();
  }

  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${tabName}"]`)?.classList.add('active');

  // Network tab uses a non-standard panel id to avoid id collision with the button.
  // When it's active, swap the paper header for a minimal author-focused header.
  const paperHeader = document.getElementById('paper-header');
  const authorHeader = document.getElementById('author-header');

  // Tabs to hide when the network view is active (they relate to a specific paper, not an author)
  const paperOnlyTabs = ['summary', 'chat', 'library', 'preferences'];

  if (tabName === 'network') {
    // Hide paper header, show author header
    if (paperHeader) paperHeader.classList.add('hidden');
    if (authorHeader && state.coauthorGraph) {
      const g = state.coauthorGraph;
      const paperCount = state.searchResults ? state.searchResults.length : 0;
      authorHeader.innerHTML = `
        <div class="author-header-inner">
          <div class="author-header-name">${escHtml(g.query_author)}</div>
          <div class="author-header-meta">${paperCount} paper${paperCount !== 1 ? 's' : ''} · Co-author Network</div>
        </div>`;
      authorHeader.classList.remove('hidden');
    }
    // Hide paper-specific tabs — they don't apply to an author view
    paperOnlyTabs.forEach(t => {
      document.querySelector(`.tab[data-tab="${t}"]`)?.classList.add('hidden');
    });
    const panel = document.getElementById('tab-panel-network');
    if (panel) { panel.classList.remove('hidden'); panel.classList.add('active'); }
    renderCoauthorNetwork(state.coauthorGraph);
  } else {
    // Restore paper header, hide author header
    if (paperHeader) paperHeader.classList.remove('hidden');
    if (authorHeader) authorHeader.classList.add('hidden');
    // Restore paper-specific tabs
    paperOnlyTabs.forEach(t => {
      document.querySelector(`.tab[data-tab="${t}"]`)?.classList.remove('hidden');
    });
    // Hide the network panel when switching away
    const netPanel = document.getElementById('tab-panel-network');
    if (netPanel) netPanel.classList.remove('active');
    document.getElementById(`tab-${tabName}`)?.classList.add('active');
  }

  if (tabName === 'library') loadLibrary();
  if (tabName === 'preferences') loadPreferences();
  if (tabName === 'chat') loadQaTools();
}

function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatDate(isoStr) {
  if (!isoStr) return '';
  try { return new Date(isoStr).toLocaleDateString(); } catch { return isoStr; }
}

function generateId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ── Event wiring ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Apply saved theme before anything renders
  initTheme();

  // Theme toggle button
  document.getElementById('theme-toggle-btn').addEventListener('click', toggleTheme);
  document.getElementById('home-btn').addEventListener('click', () => {
    window.location.href = window.location.pathname;
  });

  // Bootstrap: load library immediately for landing page + search indicators
  initApp();

  // Search mode toggle
  document.getElementById('btn-topic-mode').addEventListener('click', () => {
    state.searchMode = 'topic';
    document.getElementById('btn-topic-mode').classList.add('active');
    document.getElementById('btn-author-mode').classList.remove('active');
    document.getElementById('search-input').placeholder = 'Search arXiv papers...';
    // Don't clear the co-author graph or tab — the user may just be browsing modes.
    // The graph is cleared automatically when a new topic search returns no coauthor_graph.
  });

  document.getElementById('btn-author-mode').addEventListener('click', () => {
    state.searchMode = 'author';
    document.getElementById('btn-author-mode').classList.add('active');
    document.getElementById('btn-topic-mode').classList.remove('active');
    document.getElementById('search-input').placeholder = 'Search by author name...';
  });

  // Search
  document.getElementById('search-btn').addEventListener('click', handleSearch);
  document.getElementById('search-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleSearch();
  });

  // Tabs
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // Chat
  document.getElementById('chat-send-btn').addEventListener('click', () => {
    const input = document.getElementById('chat-input');
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    sendQuestion(q);
  });
  document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const input = e.target;
      const q = input.value.trim();
      if (!q) return;
      input.value = '';
      sendQuestion(q);
    }
  });

  document.getElementById('qa-prev-page').addEventListener('click', async () => {
    if (!state.pdfDoc || state.pdfCurrentPage <= 1) return;
    state.pdfCurrentPage -= 1;
    await renderPdfPage(state.pdfCurrentPage);
  });
  document.getElementById('qa-next-page').addEventListener('click', async () => {
    if (!state.pdfDoc || state.pdfCurrentPage >= state.pdfDoc.numPages) return;
    state.pdfCurrentPage += 1;
    await renderPdfPage(state.pdfCurrentPage);
  });
  document.getElementById('qa-zoom-out').addEventListener('click', async () => {
    if (!state.pdfDoc) return;
    state.pdfScale = Math.max(0.5, +(state.pdfScale - 0.2).toFixed(1));
    document.getElementById('qa-zoom-level').textContent = Math.round(state.pdfScale * 100) + '%';
    await renderPdfPage(state.pdfCurrentPage);
  });
  document.getElementById('qa-zoom-in').addEventListener('click', async () => {
    if (!state.pdfDoc) return;
    state.pdfScale = Math.min(3.0, +(state.pdfScale + 0.2).toFixed(1));
    document.getElementById('qa-zoom-level').textContent = Math.round(state.pdfScale * 100) + '%';
    await renderPdfPage(state.pdfCurrentPage);
  });

  document.getElementById('qa-clear-assets-btn').addEventListener('click', clearQaAssets);
  document.getElementById('qa-clear-log-btn').addEventListener('click', clearQaLog);

  document.querySelectorAll('.qa-side-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      switchQaSideTab(btn.dataset.qaside);
      // Reset badge on the Logs tab when user opens it
      if (btn.dataset.qaside === 'logs') {
        state.qaLogCount = 0;
        btn.textContent = 'Logs';
      }
    });
  });

  // Approval modal buttons
  document.getElementById('btn-approve').addEventListener('click', () => submitReview('approved'));
  document.getElementById('btn-reject').addEventListener('click', () => submitReview('rejected'));
  document.getElementById('btn-revise').addEventListener('click', () => {
    document.getElementById('revision-input').classList.remove('hidden');
    document.getElementById('btn-submit-revision').classList.remove('hidden');
    document.getElementById('btn-revise').classList.add('hidden');
  });
  document.getElementById('btn-submit-revision').addEventListener('click', () => {
    const note = document.getElementById('revision-input').value.trim();
    submitReview('revised', note);
  });

  // Close modal on backdrop click
  document.getElementById('approval-modal').addEventListener('click', (e) => {
    if (e.target === document.getElementById('approval-modal')) hideModal();
  });
});
