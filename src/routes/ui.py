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
  <div class="flex items-center justify-between gap-4 flex-wrap mb-10">
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
      <div class="flex items-center gap-2 ml-3 bg-gray-900 border border-gray-800 rounded-xl p-1.5">
        <button id="viewMem"   onclick="setView('memories')"
          class="text-xs mono px-4 py-2 rounded-lg bg-cyan-500/20 text-cyan-300 flex items-center gap-2">🧠 Memories</button>
        <button id="viewAudit" onclick="setView('audit')"
          class="text-xs mono px-4 py-2 rounded-lg text-gray-400 hover:text-gray-200 flex items-center gap-2">📜 Audit</button>
        <button id="viewPools" onclick="setView('pools')"
          class="text-xs mono px-4 py-2 rounded-lg text-gray-400 hover:text-gray-200 flex items-center gap-2">👥 Pools</button>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <input id="userInput" type="text" placeholder="user_id"
        class="mono text-sm bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 w-36 focus:outline-none focus:border-cyan-500 text-gray-200"
        onkeydown="if(event.key==='Enter') load()">
      <select id="catFilter" onchange="memPage=0;render()"
        class="mono text-sm bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 focus:outline-none focus:border-cyan-500 text-gray-200">
        <option value="">All categories</option>
        <option value="fact">fact</option>
        <option value="strategy">strategy</option>
        <option value="assumption">assumption</option>
        <option value="failure">failure</option>
      </select>
      <select id="sortBy" onchange="memPage=0;render()"
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
  <div id="memPager" class="hidden"></div>

  <!-- Audit view -->
  <div id="auditView" class="hidden">
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-3">
        <span id="auditVerify" class="text-xs mono px-2 py-1 rounded-full border border-gray-700 text-gray-400">checking integrity…</span>
        <span id="auditRetention" class="text-xs mono text-gray-500"></span>
      </div>
      <div class="flex items-center gap-3">
        <select id="auditAction" onchange="loadAudit()"
          class="mono text-xs bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-gray-200">
          <option value="">All actions</option>
          <option value="read">read</option>
          <option value="write">write</option>
          <option value="delete">delete</option>
        </select>
        <button onclick="loadAudit()" class="text-xs mono bg-gray-800 hover:bg-gray-700 border border-gray-700 px-4 py-1.5 rounded-lg text-gray-200">Refresh</button>
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
    <div id="auditPager" class="hidden"></div>
    <div id="auditEmpty" class="hidden text-center text-gray-600 py-20">
      <p class="text-4xl mb-3">📜</p>
      <p class="font-medium">No audit events yet.</p>
    </div>
  </div>

  <!-- Pools view -->
  <div id="poolsView" class="hidden">
    <div class="grid grid-cols-3 gap-6">
      <!-- LEFT: pool list + create -->
      <div class="col-span-1">
        <div class="flex items-center justify-between mb-3">
          <p class="text-xs text-gray-500 uppercase tracking-widest mono">Pools</p>
          <button onclick="document.getElementById('newPoolForm').classList.toggle('hidden')"
            class="text-xs mono text-cyan-300 hover:text-cyan-200">+ New</button>
        </div>
        <div id="newPoolForm" class="hidden bg-gray-900 border border-gray-800 rounded-xl p-3 mb-3 space-y-2">
          <input id="npId"    placeholder="pool_id (e.g. sales)" class="w-full mono text-xs bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-gray-200">
          <input id="npName"  placeholder="name (e.g. Sales Team)" class="w-full mono text-xs bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-gray-200">
          <input id="npOwner" placeholder="owner user_id" class="w-full mono text-xs bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-gray-200">
          <button onclick="createPool()" class="w-full text-xs mono bg-cyan-500 hover:bg-cyan-400 text-black font-bold rounded px-2 py-1.5">Create pool</button>
        </div>
        <div id="poolList" class="space-y-2"></div>
      </div>

      <!-- RIGHT: selected pool detail -->
      <div class="col-span-2">
        <div id="poolDetail" class="hidden">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h3 id="pdName" class="text-lg font-bold text-gray-100"></h3>
              <p id="pdMeta" class="text-xs mono text-gray-500"></p>
            </div>
            <button onclick="deletePool()" class="text-xs mono text-red-400 hover:text-red-300 border border-red-900 rounded-lg px-3 py-1.5">Delete pool</button>
          </div>

          <p class="text-xs text-gray-500 uppercase tracking-widest mono mb-3">Members (attached users)</p>
          <div class="bg-gray-900 border border-gray-800 rounded-xl p-3 mb-4 flex items-end gap-2 flex-wrap">
            <input id="amId" placeholder="user_id to attach" class="flex-1 min-w-[140px] mono text-xs bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-gray-200">
            <select id="amRole" class="mono text-xs bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-gray-200">
              <option value="reader">reader (read)</option>
              <option value="contributor">contributor (read+write)</option>
              <option value="admin">admin (manage)</option>
            </select>
            <button onclick="addMember()" class="text-xs mono bg-cyan-500 hover:bg-cyan-400 text-black font-bold rounded px-3 py-1.5">Attach</button>
          </div>
          <table class="w-full text-left">
            <thead><tr class="text-[11px] uppercase tracking-wider text-gray-500 mono border-b border-gray-800">
              <th class="px-3 py-2">User</th><th class="px-3 py-2">Role</th><th class="px-3 py-2">Read</th><th class="px-3 py-2">Write</th><th class="px-3 py-2"></th>
            </tr></thead>
            <tbody id="memberBody" class="mono text-xs text-gray-300"></tbody>
          </table>
        </div>
        <div id="poolEmpty" class="text-center text-gray-600 py-24">
          <p class="text-4xl mb-3">👥</p>
          <p class="font-medium">Select a pool, or create one.</p>
          <p class="text-xs text-gray-600 mt-2">Attach users with a role. Members' recall can auto-include their pools.</p>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- Memory detail modal (opened from an audit row's target id) -->
<div id="memModal" class="hidden fixed inset-0 z-50 flex items-center justify-center p-6"
     style="background:rgba(0,0,0,0.6)" onclick="if(event.target===this) closeMemory()">
  <div class="bg-gray-900 border border-gray-700 rounded-xl max-w-lg w-full p-6 shadow-2xl">
    <div class="flex items-center justify-between mb-4">
      <h3 id="memModalTitle" class="text-sm font-bold text-cyan-400 mono">Memory</h3>
      <button onclick="closeMemory()" class="text-gray-500 hover:text-gray-200 text-lg leading-none">✕</button>
    </div>
    <div id="memModalBody" class="text-sm text-gray-200 leading-relaxed"></div>
  </div>
</div>

<script>
  let allMemories = [];
  let agents = [];
  let activeTab = 'all';

  // ── Pagination (client-side over already-fetched data, so no page loads everything) ──
  const MEM_PAGE_SIZE = 18;
  const AUDIT_PAGE_SIZE = 25;
  let memPage = 0;
  let auditPage = 0;
  let auditEvents = [];

  // Render a Prev / page-x-of-y / Next bar into `containerId`. `onGo(newPage)` re-renders.
  function renderPager(containerId, total, page, pageSize, onGoName) {
    const el = document.getElementById(containerId);
    const pages = Math.max(1, Math.ceil(total / pageSize));
    if (total <= pageSize) { el.classList.add('hidden'); el.innerHTML = ''; return; }
    el.classList.remove('hidden');
    const from = page * pageSize + 1;
    const to   = Math.min((page + 1) * pageSize, total);
    const btn = (label, target, disabled) =>
      `<button onclick="${onGoName}(${target})" ${disabled ? 'disabled' : ''}
         class="mono text-sm px-4 py-1.5 rounded-lg border ${disabled
           ? 'border-gray-800 text-gray-700 cursor-not-allowed'
           : 'border-gray-700 text-gray-300 hover:border-cyan-500 hover:text-cyan-300'}">${label}</button>`;
    el.innerHTML = `
      <div class="flex items-center justify-center gap-5 py-6">
        ${btn('← Prev', page - 1, page === 0)}
        <span class="mono text-xs text-gray-500">${from}–${to} of ${total} · page ${page + 1}/${pages}</span>
        ${btn('Next →', page + 1, page >= pages - 1)}
      </div>`;
  }

  function goMemPage(p) { memPage = p; render(); window.scrollTo({top:0,behavior:'smooth'}); }
  function goAuditPage(p) { auditPage = p; renderAudit(auditEvents); window.scrollTo({top:0,behavior:'smooth'}); }

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
        fetch(`/memories?userId=${encodeURIComponent(uid)}&limit=500&audit=false`),
        fetch(`/agents?user_id=${encodeURIComponent(uid)}`),
      ]);

      const memData   = await memRes.json();
      const agentData = agentRes.ok ? await agentRes.json() : { agents: [] };

      allMemories = memData.memories || [];
      agents      = (agentData.agents || []).filter(a => !a.revoked_at);

      buildTabs();
      activeTab = 'all';
      memPage = 0;
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
    memPage = 0;
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
      document.getElementById('memPager').classList.add('hidden');
      return;
    }
    document.getElementById('empty').classList.add('hidden');

    // Clamp the page (filters may have shrunk the set) and show only this page's slice.
    const pageCount = Math.max(1, Math.ceil(mems.length / MEM_PAGE_SIZE));
    if (memPage >= pageCount) memPage = pageCount - 1;
    const pageMems = mems.slice(memPage * MEM_PAGE_SIZE, (memPage + 1) * MEM_PAGE_SIZE);
    renderPager('memPager', mems.length, memPage, MEM_PAGE_SIZE, 'goMemPage');

    grid.innerHTML = pageMems.map(m => {
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
    const isMem = v === 'memories', isAudit = v === 'audit', isPools = v === 'pools';
    const on = 'text-xs mono px-4 py-2 rounded-lg bg-cyan-500/20 text-cyan-300 flex items-center gap-2';
    const off = 'text-xs mono px-4 py-2 rounded-lg text-gray-400 hover:text-gray-200 flex items-center gap-2';
    document.getElementById('viewMem').className   = isMem   ? on : off;
    document.getElementById('viewAudit').className = isAudit ? on : off;
    document.getElementById('viewPools').className = isPools ? on : off;
    // memory-only chrome hidden outside the Memories view
    ['catFilter','sortBy','stats','tabsRow','grid','empty','memPager'].forEach(id =>
      document.getElementById(id).classList.toggle('hidden', !isMem));
    document.getElementById('auditView').classList.toggle('hidden', !isAudit);
    document.getElementById('poolsView').classList.toggle('hidden', !isPools);
    // Reflect the view in the URL so it's shareable/bookmarkable (preserve ?user).
    const params = new URLSearchParams(location.search);
    if (!isMem) params.set('view', v); else params.delete('view');
    const uid = document.getElementById('userInput').value.trim();
    if (uid) params.set('user', uid); else params.delete('user');
    const qs = params.toString();
    history.replaceState(null, '', qs ? '?' + qs : location.pathname);
    if (isAudit) loadAudit();
    else if (isPools) loadPools();
    else render();
  }

  async function loadAudit() {
    const uid = document.getElementById('userInput').value.trim().toLowerCase();
    const action = document.getElementById('auditAction').value;
    const qs = new URLSearchParams({ limit: '200' });
    if (uid) qs.set('userId', uid);
    if (action) qs.set('action', action);
    const body = document.getElementById('auditBody');
    body.innerHTML = '<tr><td colspan="7" class="px-4 py-8 text-center text-gray-600">Loading…</td></tr>';
    auditPage = 0;   // fresh fetch → back to first page
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
    auditEvents = events;
    const body  = document.getElementById('auditBody');
    const empty = document.getElementById('auditEmpty');
    if (!events.length) {
      body.innerHTML = '';
      empty.classList.remove('hidden');
      document.getElementById('auditPager').classList.add('hidden');
      return;
    }
    empty.classList.add('hidden');
    const pageCount = Math.max(1, Math.ceil(events.length / AUDIT_PAGE_SIZE));
    if (auditPage >= pageCount) auditPage = pageCount - 1;
    const pageEvents = events.slice(auditPage * AUDIT_PAGE_SIZE, (auditPage + 1) * AUDIT_PAGE_SIZE);
    renderPager('auditPager', events.length, auditPage, AUDIT_PAGE_SIZE, 'goAuditPage');
    body.innerHTML = pageEvents.map(e => {
      const cls = ACTION_COLORS[e.action] || 'text-gray-300 border-gray-700 bg-gray-800/40';
      const ts  = (e.ts || '').replace('T', ' ').replace(/\\.\\d+Z?$/, '');
      const uid = escHtml(e.actor_user_id);
      const link = id => `<button onclick="showMemory('${uid}',${id})" class="text-cyan-400 hover:underline">#${escHtml(id)}</button>`;
      // Single target id (write/delete/get) → one link. Otherwise pull ids out of the
      // detail blob (read/list/retrieve span several memories) → render each as a chip.
      let target = '<span class="text-gray-600">—</span>';
      if (e.target_id != null) {
        target = link(e.target_id);
      } else {
        let ids = [];
        try { ids = (JSON.parse(e.detail || '{}').ids) || []; } catch (_) {}
        if (ids.length) {
          target = ids.map(link).join('<span class="text-gray-700"> </span>');
        }
      }
      return `
      <tr class="border-b border-gray-800/60 hover:bg-gray-800/30">
        <td class="px-4 py-2 text-gray-400">${escHtml(ts)}</td>
        <td class="px-4 py-2"><span class="px-2 py-0.5 rounded-full border text-[11px] ${cls}">${escHtml(e.action)}</span></td>
        <td class="px-4 py-2 text-gray-200">${escHtml(e.operation)}</td>
        <td class="px-4 py-2 text-gray-300">${uid}</td>
        <td class="px-4 py-2 text-gray-500">${e.actor_agent_id ? escHtml(e.actor_agent_id) : '—'}</td>
        <td class="px-4 py-2 leading-relaxed">${target}</td>
        <td class="px-4 py-2 text-gray-600">${escHtml(e.source)}</td>
      </tr>`;
    }).join('');
  }

  // Click a target id in an audit row → pull up that memory's content.
  async function showMemory(userId, id) {
    const modal = document.getElementById('memModal');
    const title = document.getElementById('memModalTitle');
    const body  = document.getElementById('memModalBody');
    title.textContent = 'Memory #' + id;
    body.innerHTML = '<p class="text-gray-500 mono text-xs">Loading…</p>';
    modal.classList.remove('hidden');
    try {
      const r = await fetch(`/memory/get?userId=${encodeURIComponent(userId)}&id=${encodeURIComponent(id)}`);
      const d = await r.json();
      const m = d.memory;
      if (!m) {
        body.innerHTML = '<p class="text-gray-400">This memory no longer exists — it was deleted or is not owned by <span class="mono text-gray-300">' + escHtml(userId) + '</span>.</p>';
        return;
      }
      const nb = (d.neighbors || []).slice(0, 5);
      body.innerHTML = `
        <p class="text-gray-100 mb-4">${escHtml(m.content || '')}</p>
        <div class="flex flex-wrap gap-2 text-[11px] mono text-gray-500 mb-1">
          ${m.category   != null ? `<span class="border border-gray-700 rounded-full px-2 py-0.5">${escHtml(m.category)}</span>` : ''}
          ${m.importance != null ? `<span class="border border-gray-700 rounded-full px-2 py-0.5">importance ${escHtml(m.importance)}</span>` : ''}
          ${m.agent_id   ? `<span class="border border-purple-800 bg-purple-900/40 text-purple-300 rounded-full px-2 py-0.5">🤖 ${escHtml(m.agent_id)}</span>` : ''}
        </div>
        ${nb.length ? `<div class="mt-4 pt-3 border-t border-gray-800">
          <p class="text-[11px] uppercase tracking-wider text-gray-500 mono mb-2">Linked memories</p>
          ${nb.map(n => `<p class="text-xs text-gray-400 mb-1">• ${escHtml(n.content || '')}</p>`).join('')}
        </div>` : ''}`;
    } catch (e) {
      body.innerHTML = '<p class="text-red-400">Error: ' + escHtml(e.message) + '</p>';
    }
  }

  function closeMemory() {
    document.getElementById('memModal').classList.add('hidden');
  }
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMemory(); });

  // ── Pools (team / institutional memory management) ───────────────────────────
  let selectedPool = null;

  async function loadPools() {
    const el = document.getElementById('poolList');
    el.innerHTML = '<p class="text-xs text-gray-600 mono">Loading…</p>';
    try {
      const d = await (await fetch('/pools')).json();
      if (!d.pools.length) { el.innerHTML = '<p class="text-xs text-gray-600 mono">No pools yet.</p>'; return; }
      el.innerHTML = d.pools.map(p => `
        <button onclick="selectPool('${escHtml(p.pool_id)}')"
          class="w-full text-left bg-gray-900 border ${selectedPool===p.pool_id?'border-cyan-600':'border-gray-800'} rounded-lg px-3 py-2 hover:border-gray-600">
          <div class="text-sm text-gray-100">${escHtml(p.name || p.pool_id)}</div>
          <div class="text-[11px] mono text-gray-500">${escHtml(p.pool_id)} · owner ${escHtml(p.owner||'—')}</div>
        </button>`).join('');
    } catch (e) { el.innerHTML = `<p class="text-xs text-red-400">Error: ${escHtml(e.message)}</p>`; }
  }

  async function createPool() {
    const pool_id = document.getElementById('npId').value.trim();
    const name = document.getElementById('npName').value.trim();
    const owner = document.getElementById('npOwner').value.trim();
    if (!pool_id || !owner) { alert('pool_id and owner are required'); return; }
    const r = await fetch('/pools', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({pool_id, name, owner})});
    if (r.ok) {
      document.getElementById('npId').value = document.getElementById('npName').value = document.getElementById('npOwner').value = '';
      document.getElementById('newPoolForm').classList.add('hidden');
      await loadPools(); selectPool(pool_id);
    } else { alert('Create failed'); }
  }

  async function deletePool() {
    if (!selectedPool || !confirm(`Delete pool "${selectedPool}" and all its memories?`)) return;
    await fetch('/pools/' + encodeURIComponent(selectedPool), {method:'DELETE'});
    selectedPool = null;
    document.getElementById('poolDetail').classList.add('hidden');
    document.getElementById('poolEmpty').classList.remove('hidden');
    loadPools();
  }

  async function selectPool(id) {
    selectedPool = id;
    document.getElementById('poolEmpty').classList.add('hidden');
    document.getElementById('poolDetail').classList.remove('hidden');
    loadPools();
    try {
      const d = await (await fetch('/pools/' + encodeURIComponent(id) + '/members')).json();
      document.getElementById('pdName').textContent = id;
      document.getElementById('pdMeta').textContent = d.count + ' member(s) · namespace pool:' + id;
      renderMembers(d.members || []);
    } catch (e) { document.getElementById('memberBody').innerHTML = `<tr><td colspan="5" class="px-3 py-3 text-red-400">Error: ${escHtml(e.message)}</td></tr>`; }
  }

  function renderMembers(members) {
    const tb = document.getElementById('memberBody');
    if (!members.length) { tb.innerHTML = '<tr><td colspan="5" class="px-3 py-4 text-gray-600 text-center">No members yet — attach a user above.</td></tr>'; return; }
    const yn = b => b ? '<span class="text-green-400">✓</span>' : '<span class="text-gray-600">—</span>';
    tb.innerHTML = members.map(m => `
      <tr class="border-b border-gray-800/60">
        <td class="px-3 py-2 text-gray-200">${escHtml(m.member_id)}</td>
        <td class="px-3 py-2">${escHtml(m.role)}</td>
        <td class="px-3 py-2">${yn(m.can_read)}</td>
        <td class="px-3 py-2">${yn(m.can_write)}</td>
        <td class="px-3 py-2 text-right"><button onclick="removeMember('${escHtml(m.member_id)}')" class="text-red-400 hover:text-red-300">detach</button></td>
      </tr>`).join('');
  }

  async function addMember() {
    const member_id = document.getElementById('amId').value.trim();
    const role = document.getElementById('amRole').value;
    if (!member_id || !selectedPool) return;
    await fetch('/pools/' + encodeURIComponent(selectedPool) + '/members', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify({member_id, role})});
    document.getElementById('amId').value = '';
    selectPool(selectedPool);
  }

  async function removeMember(member_id) {
    if (!selectedPool) return;
    await fetch('/pools/' + encodeURIComponent(selectedPool) + '/members/' + encodeURIComponent(member_id), {method:'DELETE'});
    selectPool(selectedPool);
  }

  const p = new URLSearchParams(location.search);
  if (p.get('user')) {
    document.getElementById('userInput').value = p.get('user');
    load();
  }
  // Deep-link a view: /ui?view=audit or ?view=pools  (optionally with &user=…)
  if (['audit','pools'].includes(p.get('view'))) {
    setView(p.get('view'));
  }
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse)
def memory_ui():
    return HTMLResponse(content=_HTML)
