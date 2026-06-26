from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YourMemory — Memory Browser</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'DM Sans', sans-serif; background: #0d1117; color: #e6edf3; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .strength-bar { transition: width 0.4s ease; }
    .card:hover { border-color: #00D4FF44; }
    .tab { cursor: pointer; transition: all 0.15s; }
    .tab.active { background: #00D4FF18; border-color: #00D4FF66; color: #00D4FF; }
    .tab:hover:not(.active) { border-color: #30363d; background: #161b22; }
  </style>
</head>
<body class="min-h-screen p-8">
<div class="max-w-5xl mx-auto">

  <!-- Header -->
  <div class="flex items-center justify-between mb-8">
    <div class="flex items-center gap-3">
      <svg width="28" height="28" viewBox="0 0 100 100" fill="none">
        <rect x="10" y="80" width="80" height="10" rx="2" fill="#e6edf3"/>
        <rect x="25" y="60" width="50" height="10" rx="2" fill="#e6edf3"/>
        <rect x="40" y="40" width="15" height="10" rx="2" fill="#e6edf3"/>
        <rect x="60" y="40" width="5"  height="10" rx="1" fill="#00D4FF"/>
        <rect x="47.5" y="20" width="5" height="10" rx="1" fill="#00D4FF" fill-opacity="0.6"/>
      </svg>
      <span class="text-xl font-bold tracking-tight">YourMemory</span>
      <span class="text-xs mono text-gray-500 border border-gray-700 px-2 py-0.5 rounded-full">Memory Browser</span>
      <div class="flex items-center gap-1 ml-2 bg-gray-900 border border-gray-800 rounded-lg p-0.5">
        <button id="viewMem"   onclick="setView('memories')"
          class="text-xs mono px-2.5 py-1 rounded-md bg-cyan-500/20 text-cyan-300">🧠 Memories</button>
        <button id="viewAudit" onclick="setView('audit')"
          class="text-xs mono px-2.5 py-1 rounded-md text-gray-400 hover:text-gray-200">📜 Audit</button>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <input id="userInput" type="text" placeholder="user_id"
        class="mono text-sm bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 w-36 focus:outline-none focus:border-cyan-500 text-gray-200"
        onkeydown="if(event.key==='Enter') load()">
      <select id="catFilter" onchange="render()"
        class="mono text-sm bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 focus:outline-none focus:border-cyan-500 text-gray-200">
        <option value="">All categories</option>
        <option value="fact">fact</option>
        <option value="strategy">strategy</option>
        <option value="assumption">assumption</option>
        <option value="failure">failure</option>
      </select>
      <select id="sortBy" onchange="render()"
        class="mono text-sm bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 focus:outline-none focus:border-cyan-500 text-gray-200">
        <option value="strength">Sort: strength</option>
        <option value="recent">Sort: recent</option>
        <option value="recall">Sort: recall count</option>
      </select>
      <button onclick="load()"
        class="bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-sm px-4 py-1.5 rounded-lg transition-colors">
        Load
      </button>
    </div>
  </div>

  <!-- Stats bar -->
  <div id="stats" class="hidden grid grid-cols-4 gap-4 mb-6">
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p class="mono text-2xl font-bold text-cyan-400" id="statTotal">—</p>
      <p class="text-xs text-gray-500 mt-1">Total memories</p>
    </div>
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p class="mono text-2xl font-bold text-green-400" id="statStrong">—</p>
      <p class="text-xs text-gray-500 mt-1">Strong (≥ 0.5)</p>
    </div>
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p class="mono text-2xl font-bold text-yellow-400" id="statFading">—</p>
      <p class="text-xs text-gray-500 mt-1">Fading (0.05–0.5)</p>
    </div>
    <div class="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p class="mono text-2xl font-bold text-red-400" id="statCritical">—</p>
      <p class="text-xs text-gray-500 mt-1">Near prune (&lt; 0.1)</p>
    </div>
  </div>

  <!-- Agent tabs -->
  <div id="tabsRow" class="hidden mb-6">
    <p class="text-xs text-gray-500 uppercase tracking-widest mono mb-3">View</p>
    <div id="tabs" class="flex flex-wrap gap-2"></div>
  </div>

  <!-- Memory grid -->
  <div id="grid" class="space-y-3"></div>
  <div id="empty" class="hidden text-center text-gray-600 py-24">
    <p class="text-4xl mb-4">🧠</p>
    <p class="font-medium">No memories found.</p>
  </div>
  <div id="loading" class="hidden text-center text-gray-600 py-24 mono text-sm">Loading…</div>

  <!-- Audit view -->
  <div id="auditView" class="hidden">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-3">
        <span id="auditVerify" class="text-xs mono px-2 py-1 rounded-full border border-gray-700 text-gray-400">checking integrity…</span>
        <span id="auditRetention" class="text-xs mono text-gray-500"></span>
      </div>
      <div class="flex items-center gap-2">
        <select id="auditAction" onchange="loadAudit()"
          class="mono text-xs bg-gray-900 border border-gray-700 rounded-lg px-2 py-1 text-gray-200">
          <option value="">All actions</option>
          <option value="read">read</option>
          <option value="write">write</option>
          <option value="delete">delete</option>
        </select>
        <button onclick="loadAudit()" class="text-xs mono bg-gray-800 hover:bg-gray-700 border border-gray-700 px-3 py-1 rounded-lg text-gray-200">Refresh</button>
      </div>
    </div>
    <div class="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
      <table class="w-full text-left">
        <thead>
          <tr class="text-[11px] uppercase tracking-wider text-gray-500 mono border-b border-gray-800">
            <th class="px-4 py-2.5">Timestamp (UTC)</th>
            <th class="px-4 py-2.5">Action</th>
            <th class="px-4 py-2.5">Operation</th>
            <th class="px-4 py-2.5">User</th>
            <th class="px-4 py-2.5">Agent</th>
            <th class="px-4 py-2.5">Target</th>
            <th class="px-4 py-2.5">Src</th>
          </tr>
        </thead>
        <tbody id="auditBody" class="mono text-xs text-gray-300"></tbody>
      </table>
    </div>
    <div id="auditEmpty" class="hidden text-center text-gray-600 py-20">
      <p class="text-4xl mb-3">📜</p>
      <p class="font-medium">No audit events yet.</p>
    </div>
  </div>

</div>

<script>
  let allMemories = [];
  let agents = [];
  let activeTab = 'all';

  const CAT_COLORS = {
    fact:       'bg-blue-900/50 text-blue-300 border-blue-800',
    strategy:   'bg-green-900/50 text-green-300 border-green-800',
    assumption: 'bg-yellow-900/50 text-yellow-300 border-yellow-800',
    failure:    'bg-red-900/50 text-red-300 border-red-800',
  };

  function strengthColor(s) {
    if (s >= 0.7) return 'bg-green-500';
    if (s >= 0.4) return 'bg-cyan-500';
    if (s >= 0.1) return 'bg-yellow-500';
    return 'bg-red-500';
  }

  async function load() {
    const uid = document.getElementById('userInput').value.trim().toLowerCase();
    if (!uid) return;
    document.getElementById('userInput').value = uid;

    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('grid').innerHTML = '';
    document.getElementById('empty').classList.add('hidden');
    document.getElementById('stats').classList.add('hidden');
    document.getElementById('tabsRow').classList.add('hidden');

    try {
      const [memRes, agentRes] = await Promise.all([
        fetch(`/memories?userId=${encodeURIComponent(uid)}&limit=500`),
        fetch(`/agents?user_id=${encodeURIComponent(uid)}`),
      ]);

      const memData   = await memRes.json();
      const agentData = agentRes.ok ? await agentRes.json() : { agents: [] };

      allMemories = memData.memories || [];
      agents      = (agentData.agents || []).filter(a => !a.revoked_at);

      buildTabs();
      activeTab = 'all';
      render();
    } catch(e) {
      document.getElementById('grid').innerHTML =
        `<p class="text-red-400 mono text-sm text-center py-12">Error: ${e.message}</p>`;
    } finally {
      document.getElementById('loading').classList.add('hidden');
    }
  }

  function buildTabs() {
    const tabs = document.getElementById('tabs');
    const row  = document.getElementById('tabsRow');

    const defs = [
      { id: 'all',  label: 'All', icon: '🧠' },
      { id: 'user', label: 'User', icon: '👤' },
      ...agents.map(a => ({ id: a.agent_id, label: a.agent_id, icon: '🤖' })),
    ];

    tabs.innerHTML = defs.map(t => `
      <button
        data-tab-id="${escHtml(t.id)}"
        id="tab-${escHtml(t.id)}"
        class="tab border border-gray-800 rounded-lg px-3 py-1.5 text-sm mono text-gray-400 flex items-center gap-1.5"
      >${t.icon} ${escHtml(t.label)}</button>
    `).join('');

    tabs.querySelectorAll('.tab[data-tab-id]').forEach(btn => {
      btn.addEventListener('click', () => setTab(btn.dataset.tabId));
    });

    row.classList.remove('hidden');
  }

  function setTab(id) {
    activeTab = id;
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    const el = document.getElementById('tab-' + id);
    if (el) el.classList.add('active');
    render();
  }

  function render() {
    const cat    = document.getElementById('catFilter').value;
    const sortBy = document.getElementById('sortBy').value;

    let mems = allMemories;

    // Tab filter
    if (activeTab === 'user') {
      mems = mems.filter(m => !m.agent_id || m.agent_id === 'user');
    } else if (activeTab !== 'all') {
      mems = mems.filter(m => m.agent_id === activeTab);
    }

    // Category filter
    if (cat) mems = mems.filter(m => m.category === cat);

    // Sort
    if (sortBy === 'strength')     mems.sort((a,b) => b.strength - a.strength);
    else if (sortBy === 'recent')  mems.sort((a,b) => new Date(b.last_accessed_at) - new Date(a.last_accessed_at));
    else if (sortBy === 'recall')  mems.sort((a,b) => b.recall_count - a.recall_count);

    // Stats (always over full set for the active tab)
    const base     = activeTab === 'all'  ? allMemories
                   : activeTab === 'user' ? allMemories.filter(m => !m.agent_id || m.agent_id === 'user')
                   : allMemories.filter(m => m.agent_id === activeTab);
    const strong   = base.filter(m => m.strength >= 0.5).length;
    const fading   = base.filter(m => m.strength >= 0.05 && m.strength < 0.5).length;
    const critical = base.filter(m => m.strength < 0.1).length;

    document.getElementById('statTotal').textContent    = base.length;
    document.getElementById('statStrong').textContent   = strong;
    document.getElementById('statFading').textContent   = fading;
    document.getElementById('statCritical').textContent = critical;
    document.getElementById('stats').classList.remove('hidden');
    document.getElementById('stats').classList.add('grid');

    // Activate first tab on first render
    if (!document.querySelector('.tab.active')) setTab('all');

    const grid = document.getElementById('grid');
    if (!mems.length) {
      grid.innerHTML = '';
      document.getElementById('empty').classList.remove('hidden');
      return;
    }
    document.getElementById('empty').classList.add('hidden');

    grid.innerHTML = mems.map(m => {
      const pct     = Math.round(m.strength * 100);
      const barCol  = strengthColor(m.strength);
      const catCls  = CAT_COLORS[m.category] || 'bg-gray-800 text-gray-300 border-gray-700';
      const accessed = m.last_accessed_at ? m.last_accessed_at.split('T')[0] : '—';
      const agentBadge = m.agent_id && m.agent_id !== 'user'
        ? `<span class="text-xs border border-purple-800 bg-purple-900/50 text-purple-300 px-2 py-0.5 rounded-full mono">🤖 ${escHtml(m.agent_id)}</span>`
        : `<span class="text-xs border border-gray-700 bg-gray-800/50 text-gray-500 px-2 py-0.5 rounded-full mono">👤 user</span>`;
      return `
      <div class="card bg-gray-900 border border-gray-800 rounded-xl p-5 transition-colors duration-200">
        <div class="flex items-start justify-between gap-4 mb-3">
          <p class="text-sm text-gray-100 leading-relaxed flex-1">${escHtml(m.content)}</p>
          <div class="flex items-center gap-2 shrink-0 flex-wrap justify-end">
            <span class="mono text-xs font-bold ${pct >= 50 ? 'text-green-400' : pct >= 10 ? 'text-yellow-400' : 'text-red-400'}">${pct}%</span>
            <span class="text-xs border px-2 py-0.5 rounded-full mono ${catCls}">${m.category}</span>
            ${agentBadge}
          </div>
        </div>
        <div class="h-1.5 bg-gray-800 rounded-full overflow-hidden mb-3">
          <div class="h-full ${barCol} rounded-full strength-bar" style="width:${pct}%"></div>
        </div>
        <div class="flex items-center gap-4 text-[11px] mono text-gray-600">
          <span>id: ${m.id}</span>
          <span>importance: ${m.importance}</span>
          <span>recalls: ${m.recall_count}</span>
          <span>last accessed: ${accessed}</span>
        </div>
      </div>`;
    }).join('');
  }

  function escHtml(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  // ── Audit view ─────────────────────────────────────────────────────────────
  let currentView = 'memories';
  const ACTION_COLORS = {
    read:   'text-blue-300 border-blue-800 bg-blue-900/40',
    write:  'text-green-300 border-green-800 bg-green-900/40',
    delete: 'text-red-300 border-red-800 bg-red-900/40',
    admin:  'text-purple-300 border-purple-800 bg-purple-900/40',
  };

  function setView(v) {
    currentView = v;
    const isAudit = v === 'audit';
    document.getElementById('viewMem').className =
      'text-xs mono px-2.5 py-1 rounded-md ' + (isAudit ? 'text-gray-400 hover:text-gray-200' : 'bg-cyan-500/20 text-cyan-300');
    document.getElementById('viewAudit').className =
      'text-xs mono px-2.5 py-1 rounded-md ' + (isAudit ? 'bg-cyan-500/20 text-cyan-300' : 'text-gray-400 hover:text-gray-200');
    // toggle memory-only chrome
    ['catFilter','sortBy'].forEach(id => document.getElementById(id).classList.toggle('hidden', isAudit));
    ['stats','tabsRow','grid','empty'].forEach(id => document.getElementById(id).classList.toggle('hidden', isAudit));
    document.getElementById('auditView').classList.toggle('hidden', !isAudit);
    if (isAudit) loadAudit(); else render();
  }

  async function loadAudit() {
    const uid = document.getElementById('userInput').value.trim().toLowerCase();
    const action = document.getElementById('auditAction').value;
    const qs = new URLSearchParams({ limit: '200' });
    if (uid) qs.set('userId', uid);
    if (action) qs.set('action', action);
    const body = document.getElementById('auditBody');
    body.innerHTML = '<tr><td colspan="7" class="px-4 py-8 text-center text-gray-600">Loading…</td></tr>';
    try {
      const [aRes, vRes] = await Promise.all([
        fetch('/audit?' + qs.toString()),
        fetch('/audit/verify'),
      ]);
      const data = await aRes.json();
      const v = vRes.ok ? await vRes.json() : { ok: false };
      renderVerify(v);
      document.getElementById('auditRetention').textContent =
        'retention: ' + (data.retention_days || 90) + ' days';
      renderAudit(data.events || []);
    } catch (e) {
      body.innerHTML = `<tr><td colspan="7" class="px-4 py-8 text-center text-red-400">Error: ${escHtml(e.message)}</td></tr>`;
    }
  }

  function renderVerify(v) {
    const el = document.getElementById('auditVerify');
    if (v && v.ok) {
      el.className = 'text-xs mono px-2 py-1 rounded-full border border-green-800 bg-green-900/40 text-green-300';
      el.textContent = '✓ chain verified · ' + (v.rows || 0) + ' rows';
    } else {
      el.className = 'text-xs mono px-2 py-1 rounded-full border border-red-800 bg-red-900/40 text-red-300';
      el.textContent = '✗ tamper detected' + (v && v.broken_at ? ' @ id ' + v.broken_at : '');
    }
  }

  function renderAudit(events) {
    const body  = document.getElementById('auditBody');
    const empty = document.getElementById('auditEmpty');
    if (!events.length) {
      body.innerHTML = '';
      empty.classList.remove('hidden');
      return;
    }
    empty.classList.add('hidden');
    body.innerHTML = events.map(e => {
      const cls = ACTION_COLORS[e.action] || 'text-gray-300 border-gray-700 bg-gray-800/40';
      const ts  = (e.ts || '').replace('T', ' ').replace(/\\.\\d+Z?$/, '');
      return `
      <tr class="border-b border-gray-800/60 hover:bg-gray-800/30">
        <td class="px-4 py-2 text-gray-400">${escHtml(ts)}</td>
        <td class="px-4 py-2"><span class="px-2 py-0.5 rounded-full border text-[11px] ${cls}">${escHtml(e.action)}</span></td>
        <td class="px-4 py-2 text-gray-200">${escHtml(e.operation)}</td>
        <td class="px-4 py-2 text-gray-300">${escHtml(e.actor_user_id)}</td>
        <td class="px-4 py-2 text-gray-500">${e.actor_agent_id ? escHtml(e.actor_agent_id) : '—'}</td>
        <td class="px-4 py-2 text-gray-500">${e.target_id == null ? '—' : escHtml(e.target_id)}</td>
        <td class="px-4 py-2 text-gray-600">${escHtml(e.source)}</td>
      </tr>`;
    }).join('');
  }

  const p = new URLSearchParams(location.search);
  if (p.get('user')) {
    document.getElementById('userInput').value = p.get('user');
    load();
  }
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse)
def memory_ui():
    return HTMLResponse(content=_HTML)
