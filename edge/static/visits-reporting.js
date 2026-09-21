/**
 * Inbound Surveillance — Visits & Customer Attendance Intelligence Component
 * Modular presentation component for customer visits, multi-angle appearance tracking,
 * entry/exit dwell analytics, avatar preview, and owner controls.
 * Complies with SYSTEM_ARCHITECTURE_LAWS.md (Law 2, Law 4).
 */

(function () {
  'use strict';

  let currentFilter = 'today';
  let currentSort = 'recent';
  let searchQuery = '';
  let cachedData = null;
  let targetContainer = null;

  function esc(str) {
    if (str == null) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function formatDwell(seconds) {
    if (seconds == null || isNaN(seconds) || seconds <= 0) return '—';
    const s = Math.round(Number(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const remS = s % 60;
    if (h > 0) return `${h}h ${m}m ${remS}s`;
    if (m > 0) return `${m}m ${remS}s`;
    return `${remS}s`;
  }

  function formatTime(isoStr) {
    if (!isoStr) return '—';
    try {
      const d = new Date(isoStr);
      if (isNaN(d.getTime())) return isoStr;
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (_) {
      return isoStr;
    }
  }

  function formatDate(isoStr) {
    if (!isoStr) return '—';
    try {
      const d = new Date(isoStr);
      if (isNaN(d.getTime())) return isoStr;
      return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + formatTime(isoStr);
    } catch (_) {
      return isoStr;
    }
  }

  async function fetchDetailedReport() {
    try {
      const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
      const timeoutId = controller ? setTimeout(() => controller.abort(), 6000) : null;
      const res = await fetch(`/api/workplace/visits/detailed?range=${encodeURIComponent(currentFilter)}`, {
        cache: 'no-store',
        signal: controller ? controller.signal : undefined,
      });
      if (timeoutId) clearTimeout(timeoutId);
      if (res.ok) {
        const json = await res.json();
        if (json && typeof json === 'object') return json;
      }
    } catch (err) {
      console.warn('[VisitsReporting] Fetch failed:', err);
    }
    return {
      summary: {
        total_unique: 0,
        today_unique: 0,
        week_unique: 0,
        today_visits: 0,
        week_visits: 0,
        total_visits: 0,
        avg_dwell_seconds: 0,
        returning_rate: 0,
        active_now_count: 0,
      },
      visitors: [],
      recent_visits: [],
      open_sessions: [],
    };
  }

  async function resetAllVisits() {
    if (!confirm('⚠️ ARE YOU SURE YOU WANT TO RESET ALL VISITS?\n\nThis will permanently remove all visitor histories, stay records, and attendance data from the system.')) {
      return;
    }
    try {
      const res = await fetch('/api/workplace/visits/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (data.ok) {
        alert('All visits have been successfully reset.');
        cachedData = null;
        if (targetContainer) render(targetContainer);
      } else {
        alert('Failed to reset visits: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      alert('Network error while resetting visits: ' + err.message);
    }
  }

  async function editAlias(subjectId, currentAlias) {
    const newName = prompt('Enter a friendly customer name or label (e.g. "Alice (VIP)", "John Regular"):', currentAlias || '');
    if (newName === null) return;
    try {
      const res = await fetch('/api/workplace/visits/alias', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject_id: subjectId, alias: newName.trim() }),
      });
      const data = await res.json();
      if (data.ok) {
        cachedData = null;
        if (targetContainer) render(targetContainer);
      } else {
        alert('Failed to save alias: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      alert('Network error while saving alias: ' + err.message);
    }
  }

  function render(container) {
    if (!container) return;
    targetContainer = container;

    // Immediate responsive skeleton so container is never a blank void
    if (!cachedData) {
      container.innerHTML = `
        <div class="space-y-4 animate-pulse">
          <div class="h-16 bg-surface-container rounded-xl border border-outline-variant/40"></div>
          <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
            <div class="h-20 bg-surface-container rounded-xl border border-outline-variant/40"></div>
            <div class="h-20 bg-surface-container rounded-xl border border-outline-variant/40"></div>
            <div class="h-20 bg-surface-container rounded-xl border border-outline-variant/40"></div>
            <div class="h-20 bg-surface-container rounded-xl border border-outline-variant/40"></div>
            <div class="h-20 bg-surface-container rounded-xl border border-outline-variant/40"></div>
          </div>
          <div class="h-64 bg-surface-container-low rounded-xl border border-outline-variant/40"></div>
        </div>
      `;
    }

    fetchDetailedReport()
      .then((data) => {
        cachedData = data;
        renderDOM(container, data);
      })
      .catch((err) => {
        console.error('[VisitsReporting] Render error:', err);
        container.innerHTML = `
          <div class="p-8 text-center rounded-xl bg-surface-container border border-outline-variant space-y-3">
            <p class="text-sm font-semibold text-error">Unable to load visits report</p>
            <p class="text-xs text-on-surface-variant">${esc(err.message || 'Network error')}</p>
            <button id="btn-retry-visits" class="px-4 py-2 text-xs font-medium rounded-lg bg-surface-container-high border border-outline-variant text-on-surface hover:bg-surface-container-highest transition-colors">
              Retry Loading
            </button>
          </div>
        `;
        const retryBtn = container.querySelector('#btn-retry-visits');
        if (retryBtn) retryBtn.addEventListener('click', () => render(container));
      });
  }

  function renderDOM(container, data) {
    const summary = data.summary || {};
    const visitors = data.visitors || [];
    const openSessions = data.open_sessions || [];

    // Filter visitors by search
    let filteredVisitors = visitors.filter((v) => {
      if (!searchQuery) return true;
      const q = searchQuery.toLowerCase();
      const sid = (v.subject_id || '').toLowerCase();
      const alias = (v.alias || '').toLowerCase();
      return sid.includes(q) || alias.includes(q);
    });

    // Sort visitors
    if (currentSort === 'recent') {
      filteredVisitors.sort((a, b) => (b.last_seen_at || '').localeCompare(a.last_seen_at || ''));
    } else if (currentSort === 'frequency') {
      filteredVisitors.sort((a, b) => (b.total_visits || 0) - (a.total_visits || 0));
    } else if (currentSort === 'dwell') {
      filteredVisitors.sort((a, b) => (b.avg_dwell_seconds || 0) - (a.avg_dwell_seconds || 0));
    }

    let html = `
      <div class="space-y-5">
        <!-- Controls & Filter Header -->
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-surface-container border border-outline-variant">
          <div>
            <div class="flex items-center gap-2">
              <span class="inline-block w-2.5 h-2.5 rounded-full bg-primary animate-pulse"></span>
              <h2 class="text-base font-semibold text-on-surface">Customer Attendance & Visit Analytics</h2>
              <span class="px-2 py-0.5 text-[11px] font-mono rounded bg-primary/10 text-primary border border-primary/20">Venue Mode</span>
            </div>
            <p class="text-xs text-on-surface-variant mt-1">Multi-angle appearance matching, dwell times, and individual customer profiles.</p>
          </div>
          <div class="flex items-center gap-2">
            <button id="btn-reset-all-visits" class="px-3 py-1.5 text-xs font-medium rounded-lg border border-error/40 text-error hover:bg-error/10 transition-colors flex items-center gap-1.5">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
              Reset All Visits
            </button>
            <button id="btn-refresh-visits" class="px-3 py-1.5 text-xs font-medium rounded-lg bg-surface-container-high border border-outline-variant text-on-surface hover:bg-surface-container-highest transition-colors flex items-center gap-1.5">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
              Refresh
            </button>
          </div>
        </div>

        <!-- 5-Card KPI Grid -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div class="rounded-xl border border-outline-variant bg-surface-container p-3">
            <div class="flex items-center justify-between">
              <p class="text-xs text-on-surface-variant">Today Unique</p>
              <span class="text-[10px] font-mono text-primary bg-primary/10 px-1.5 py-0.5 rounded">Active</span>
            </div>
            <p class="font-mono-data text-xl text-primary font-bold mt-1">${summary.today_unique || 0}</p>
            <p class="text-[11px] text-on-surface-variant mt-0.5">${summary.today_visits || 0} visits total</p>
          </div>

          <div class="rounded-xl border border-outline-variant bg-surface-container p-3">
            <p class="text-xs text-on-surface-variant">7-Day Unique</p>
            <p class="font-mono-data text-xl text-on-surface font-bold mt-1">${summary.week_unique || 0}</p>
            <p class="text-[11px] text-on-surface-variant mt-0.5">${summary.week_visits || 0} visits past week</p>
          </div>

          <div class="rounded-xl border border-outline-variant bg-surface-container p-3">
            <p class="text-xs text-on-surface-variant">Average Stay</p>
            <p class="font-mono-data text-xl text-on-surface font-bold mt-1">${formatDwell(summary.avg_dwell_seconds)}</p>
            <p class="text-[11px] text-on-surface-variant mt-0.5">Per completed visit</p>
          </div>

          <div class="rounded-xl border border-outline-variant bg-surface-container p-3">
            <p class="text-xs text-on-surface-variant">Returning Rate</p>
            <p class="font-mono-data text-xl text-on-surface font-bold mt-1">${summary.returning_rate || 0}%</p>
            <p class="text-[11px] text-on-surface-variant mt-0.5">Customers with >1 visit</p>
          </div>

          <div class="rounded-xl border border-primary/30 bg-primary/5 p-3 col-span-2 md:col-span-1">
            <div class="flex items-center gap-1.5">
              <span class="w-2 h-2 rounded-full bg-primary animate-ping"></span>
              <p class="text-xs font-medium text-primary">In Venue Now</p>
            </div>
            <p class="font-mono-data text-xl text-primary font-bold mt-1">${openSessions.length}</p>
            <p class="text-[11px] text-primary/80 mt-0.5">Active inside zones</p>
          </div>
        </div>

        <!-- Active In-Venue Visitors Cards -->
        ${
          openSessions.length > 0
            ? `
          <div class="rounded-xl border border-primary/20 bg-surface-container-low p-4">
            <h3 class="text-sm font-semibold text-on-surface mb-3 flex items-center gap-2">
              <span class="w-2 h-2 rounded-full bg-primary"></span>
              Currently Inside Venue (${openSessions.length})
            </h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
              ${openSessions
                .map((session) => {
                  const sid = esc(session.subject_id || 'visitor');
                  const zname = esc(session.zone_name || session.zone_id || 'Zone');
                  const started = formatTime(session.started_at);
                  const dwell = formatDwell(session.dwell_seconds);
                  const avatarUrl = session.avatar_url || `/api/workplace/avatar/${sid}`;
                  return `
                    <div class="rounded-lg border border-outline-variant bg-surface-container p-3 flex items-center gap-3">
                      <img src="${avatarUrl}" class="w-12 h-12 rounded-lg object-cover bg-surface-container-highest border border-outline-variant" onerror="this.src='/favicon.png';this.onerror=null;" alt="Avatar">
                      <div class="flex-1 min-w-0">
                        <div class="flex items-center justify-between">
                          <p class="text-sm font-medium text-on-surface truncate">${sid}</p>
                          <span class="px-1.5 py-0.5 text-[10px] font-mono rounded bg-primary/20 text-primary">Active</span>
                        </div>
                        <p class="text-xs text-on-surface-variant truncate">In ${zname}</p>
                        <p class="font-mono-data text-xs text-primary font-semibold mt-0.5">⏱️ ${dwell} <span class="text-[10px] text-on-surface-variant font-normal">(since ${started})</span></p>
                      </div>
                    </div>
                  `;
                })
                .join('')}
            </div>
          </div>
        `
            : ''
        }

        <!-- Directory & Filter Bar -->
        <div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <h3 class="text-sm font-semibold text-on-surface">Customer Profiles & Visit History</h3>
            <div class="flex items-center gap-2 flex-wrap">
              <input
                id="inp-visits-search"
                type="text"
                value="${esc(searchQuery)}"
                placeholder="Search visitor ID or name..."
                class="px-3 py-1.5 text-xs rounded-lg bg-surface-container border border-outline-variant text-on-surface placeholder-on-surface-variant focus:outline-none focus:border-primary w-48"
              >
              <select id="sel-visits-sort" class="px-2.5 py-1.5 text-xs rounded-lg bg-surface-container border border-outline-variant text-on-surface focus:outline-none focus:border-primary">
                <option value="recent" ${currentSort === 'recent' ? 'selected' : ''}>Sort: Most Recent</option>
                <option value="frequency" ${currentSort === 'frequency' ? 'selected' : ''}>Sort: Most Frequent</option>
                <option value="dwell" ${currentSort === 'dwell' ? 'selected' : ''}>Sort: Longest Stay</option>
              </select>
            </div>
          </div>

          ${
            filteredVisitors.length === 0
              ? `
            <div class="text-center py-8 text-on-surface-variant">
              <p class="text-sm">No visitor records found.</p>
              <p class="text-xs mt-1">Visitors will automatically appear here once detected on camera.</p>
            </div>
          `
              : `
            <div class="overflow-x-auto">
              <table class="w-full text-left text-xs text-on-surface border-collapse">
                <thead>
                  <tr class="border-b border-outline-variant text-on-surface-variant">
                    <th class="py-2.5 px-3 font-medium">Customer Profile</th>
                    <th class="py-2.5 px-3 font-medium">Status</th>
                    <th class="py-2.5 px-3 font-medium">Frequency</th>
                    <th class="py-2.5 px-3 font-medium">Stay Duration</th>
                    <th class="py-2.5 px-3 font-medium">First & Last Seen</th>
                    <th class="py-2.5 px-3 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-outline-variant/60">
                  ${filteredVisitors
                    .map((v) => {
                      const sid = esc(v.subject_id || 'visitor');
                      const alias = v.alias ? esc(v.alias) : '';
                      const avatarUrl = v.avatar_path || `/api/workplace/avatar/${sid}`;
                      const isOnline = Boolean(v.is_active_now);
                      const totalVisits = Number(v.total_visits || 0);
                      const todayVisits = Number(v.today_visits || 0);
                      const weekVisits = Number(v.week_visits || 0);
                      const avgDwell = formatDwell(v.avg_dwell_seconds);
                      const lastDwell = formatDwell(v.last_dwell_seconds);
                      const firstSeen = formatDate(v.first_seen_at);
                      const lastSeen = formatDate(v.last_seen_at);

                      return `
                        <tr class="hover:bg-surface-container/60 transition-colors">
                          <td class="py-2.5 px-3">
                            <div class="flex items-center gap-2.5">
                              <img src="${avatarUrl}" class="w-9 h-9 rounded-lg object-cover bg-surface-container-highest border border-outline-variant" onerror="this.src='/favicon.png';this.onerror=null;" alt="Avatar">
                              <div>
                                <div class="flex items-center gap-1.5">
                                  <p class="font-medium text-on-surface">${alias || sid}</p>
                                  ${alias ? `<span class="text-[10px] text-on-surface-variant font-mono">(${sid})</span>` : ''}
                                </div>
                                <p class="text-[11px] text-on-surface-variant font-mono">${v.first_seen_at ? 'Enrolled ' + formatDate(v.first_seen_at) : 'Active'}</p>
                              </div>
                            </div>
                          </td>
                          <td class="py-2.5 px-3">
                            ${
                              isOnline
                                ? `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-primary/10 text-primary border border-primary/20"><span class="w-1.5 h-1.5 rounded-full bg-primary"></span>In Venue</span>`
                                : `<span class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-surface-container-highest text-on-surface-variant">Departed</span>`
                            }
                          </td>
                          <td class="py-2.5 px-3">
                            <p class="font-mono-data font-semibold text-on-surface">${totalVisits} <span class="font-normal text-[11px] text-on-surface-variant">visits</span></p>
                            <p class="text-[11px] text-on-surface-variant mt-0.5">${todayVisits} today · ${weekVisits} this wk</p>
                          </td>
                          <td class="py-2.5 px-3">
                            <p class="font-mono-data text-on-surface font-medium">Avg: ${avgDwell}</p>
                            <p class="text-[11px] text-on-surface-variant font-mono mt-0.5">Last: ${lastDwell}</p>
                          </td>
                          <td class="py-2.5 px-3 font-mono text-[11px] text-on-surface-variant">
                            <p>Last: ${lastSeen}</p>
                          </td>
                          <td class="py-2.5 px-3 text-right">
                            <button class="btn-edit-alias px-2.5 py-1 text-xs rounded border border-outline-variant hover:border-primary text-on-surface hover:text-primary transition-colors" data-id="${sid}" data-alias="${alias}">
                              ${alias ? 'Edit Name' : '+ Set Name'}
                            </button>
                          </td>
                        </tr>
                      `;
                    })
                    .join('')}
                </tbody>
              </table>
            </div>
          `
          }
        </div>
      </div>
    `;

    container.innerHTML = html;

    // Attach Event Handlers
    const resetBtn = container.querySelector('#btn-reset-all-visits');
    if (resetBtn) {
      resetBtn.addEventListener('click', resetAllVisits);
    }

    const refreshBtn = container.querySelector('#btn-refresh-visits');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => render(container));
    }

    const searchInput = container.querySelector('#inp-visits-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderDOM(container, cachedData);
      });
    }

    const sortSelect = container.querySelector('#sel-visits-sort');
    if (sortSelect) {
      sortSelect.addEventListener('change', (e) => {
        currentSort = e.target.value;
        renderDOM(container, cachedData);
      });
    }

    const editBtns = container.querySelectorAll('.btn-edit-alias');
    editBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-id');
        const currentA = btn.getAttribute('data-alias');
        editAlias(id, currentA);
      });
    });
  }

  window.VisitsReportingComponent = {
    render: render,
  };
})();
