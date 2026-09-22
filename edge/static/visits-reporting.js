/**
 * Inbound Surveillance — Visits & Customer Attendance Intelligence Component
 * Modular presentation component for customer visits, multi-angle appearance tracking,
 * entry/exit dwell analytics, avatar preview, and owner controls.
 * Complies with SYSTEM_ARCHITECTURE_LAWS.md (Law 2, Law 4).
 */

(function () {
  'use strict';

  let currentFilter = 'today';
  let currentDateFilter = null; // 'YYYY-MM-DD' or null
  let currentSort = 'recent';
  let searchQuery = '';
  let cachedData = null;
  let cachedCalendar = [];
  let calendarMonth = null; // 'YYYY-MM'
  let targetContainer = null;
  let pollTimer = null;
  let liveClockTimer = null;
  let isFetching = false;
  let showCalendar = false;

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

  function formatDateShort(dateStr) {
    if (!dateStr) return '—';
    try {
      const parts = dateStr.split('-');
      if (parts.length === 3) {
        const d = new Date(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]));
        return d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
      }
      return dateStr;
    } catch (_) {
      return dateStr;
    }
  }

  function getMonthYearHeader(monthStr) {
    try {
      const parts = (monthStr || '').split('-');
      const y = parts[0] ? Number(parts[0]) : new Date().getFullYear();
      const m = parts[1] ? Number(parts[1]) - 1 : new Date().getMonth();
      const d = new Date(y, m, 1);
      return d.toLocaleDateString([], { month: 'long', year: 'numeric' });
    } catch (_) {
      return monthStr || 'Current Month';
    }
  }

  async function fetchDetailedReport() {
    try {
      const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
      const timeoutId = controller ? setTimeout(() => controller.abort(), 6000) : null;
      let url = `/api/workplace/visits/detailed?range=${encodeURIComponent(currentFilter)}`;
      if (currentDateFilter) {
        url += `&date=${encodeURIComponent(currentDateFilter)}`;
      }
      const res = await fetch(url, {
        cache: 'no-store',
        signal: controller ? controller.signal : undefined,
      });
      if (timeoutId) clearTimeout(timeoutId);
      if (res.ok) {
        const json = await res.json();
        if (json && typeof json === 'object') return json;
      }
    } catch (err) {
      console.warn('[VisitsReporting] Fetch detailed failed:', err);
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
        date_filter: currentDateFilter,
        date_unique: 0,
        date_visits: 0,
      },
      visitors: [],
      recent_visits: [],
      open_sessions: [],
    };
  }

  async function fetchCalendarSummary(month) {
    try {
      const m = month || calendarMonth || new Date().toISOString().substring(0, 7);
      const res = await fetch(`/api/workplace/visits/calendar?month=${encodeURIComponent(m)}`, {
        cache: 'no-store',
      });
      if (res.ok) {
        const json = await res.json();
        if (json && json.ok && Array.isArray(json.dates)) {
          cachedCalendar = json.dates;
          return cachedCalendar;
        }
      }
    } catch (err) {
      console.warn('[VisitsReporting] Fetch calendar failed:', err);
    }
    return [];
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
        cachedCalendar = [];
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
        if (cachedData && Array.isArray(cachedData.visitors)) {
          const v = cachedData.visitors.find((x) => x.subject_id === subjectId);
          if (v) v.alias = newName.trim();
        }
        if (targetContainer) renderDOM(targetContainer, cachedData, true);
      } else {
        alert('Failed to save alias: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      alert('Network error while saving alias: ' + err.message);
    }
  }

  async function executeMerge(sourceId, targetId) {
    if (!sourceId || !targetId || sourceId === targetId) {
      alert('Please select a valid profile to merge into.');
      return;
    }
    try {
      const res = await fetch('/api/workplace/visits/merge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_id: sourceId, target_id: targetId }),
      });
      const data = await res.json();
      if (data.ok) {
        closeModals();
        if (targetContainer) render(targetContainer);
      } else {
        alert('Failed to merge profiles: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      alert('Network error while merging profiles: ' + err.message);
    }
  }

  function updateLiveDwellClocks() {
    if (!targetContainer) return;
    const dwellEls = targetContainer.querySelectorAll('[data-live-started]');
    const now = Date.now();
    dwellEls.forEach((el) => {
      const iso = el.getAttribute('data-live-started');
      if (!iso) return;
      const started = new Date(iso).getTime();
      if (isNaN(started)) return;
      const diffSec = Math.max(0, Math.floor((now - started) / 1000));
      el.textContent = `⏱️ ${formatDwell(diffSec)}`;
    });
  }

  async function pollUpdates() {
    if (isFetching || !targetContainer || !document.body.contains(targetContainer)) return;
    if (targetContainer.closest('.hidden')) return;

    isFetching = true;
    try {
      const data = await fetchDetailedReport();
      cachedData = data;
      renderDOM(targetContainer, data, true);
    } catch (e) {
      console.warn('[VisitsReporting] Polling error:', e);
    } finally {
      isFetching = false;
    }
  }

  function startTimers() {
    if (!pollTimer) {
      pollTimer = setInterval(pollUpdates, 3000);
    }
    if (!liveClockTimer) {
      liveClockTimer = setInterval(updateLiveDwellClocks, 1000);
    }
  }

  function openSnapshotModal(subjectId) {
    if (!cachedData) return;
    const visitor = (cachedData.visitors || []).find((v) => v.subject_id === subjectId) || { subject_id: subjectId };
    const modal = document.getElementById('visit-snapshot-modal');
    if (!modal) return;

    const sid = visitor.subject_id || subjectId;
    const alias = visitor.alias || '';
    const avatarUrl = visitor.avatar_path || `/api/workplace/avatar/${sid}`;
    const isOnline = Boolean(visitor.is_active_now);

    const imgEl = modal.querySelector('#snapshot-modal-img');
    if (imgEl) {
      imgEl.src = avatarUrl;
      imgEl.onerror = () => { imgEl.src = '/favicon.png'; };
    }

    const titleEl = modal.querySelector('#snapshot-modal-title');
    if (titleEl) titleEl.textContent = alias || sid;

    const subEl = modal.querySelector('#snapshot-modal-sub');
    if (subEl) subEl.textContent = alias ? `ID: ${sid}` : 'Customer ID';

    const statusEl = modal.querySelector('#snapshot-modal-status');
    if (statusEl) {
      statusEl.innerHTML = isOnline
        ? `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-primary/20 text-primary border border-primary/30"><span class="w-2 h-2 rounded-full bg-primary animate-ping"></span>Currently In Venue</span>`
        : `<span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-surface-container-highest text-on-surface-variant">Departed</span>`;
    }

    const firstSeenEl = modal.querySelector('#snapshot-modal-first-seen');
    if (firstSeenEl) firstSeenEl.textContent = formatDate(visitor.first_seen_at);

    const lastSeenEl = modal.querySelector('#snapshot-modal-last-seen');
    if (lastSeenEl) lastSeenEl.textContent = formatDate(visitor.last_seen_at);

    const visitsEl = modal.querySelector('#snapshot-modal-visits');
    if (visitsEl) visitsEl.textContent = `${visitor.total_visits || 1} total visits`;

    const dwellEl = modal.querySelector('#snapshot-modal-dwell');
    if (dwellEl) dwellEl.textContent = `Avg: ${formatDwell(visitor.avg_dwell_seconds)} · Total: ${formatDwell(visitor.total_dwell_seconds)}`;

    const editBtn = modal.querySelector('#snapshot-modal-btn-edit');
    if (editBtn) {
      editBtn.onclick = () => {
        closeModals();
        editAlias(sid, alias);
      };
    }

    const mergeBtn = modal.querySelector('#snapshot-modal-btn-merge');
    if (mergeBtn) {
      mergeBtn.onclick = () => {
        closeModals();
        openMergeModal(sid);
      };
    }

    modal.classList.remove('hidden');
  }

  function openMergeModal(sourceId, preselectedTargetId = null) {
    if (!cachedData) return;
    const visitors = cachedData.visitors || [];
    const sourceVisitor = visitors.find((v) => v.subject_id === sourceId) || { subject_id: sourceId };
    const modal = document.getElementById('visit-merge-modal');
    if (!modal) return;

    const sid = sourceVisitor.subject_id;
    const alias = sourceVisitor.alias || '';
    const avatarUrl = sourceVisitor.avatar_path || `/api/workplace/avatar/${sid}`;
    const visits = sourceVisitor.total_visits || 1;

    const srcCard = modal.querySelector('#merge-source-card');
    if (srcCard) {
      srcCard.innerHTML = `
        <div class="flex items-center gap-3 p-3 rounded-xl bg-surface-container border border-outline-variant">
          <img src="${avatarUrl}" class="w-12 h-12 rounded-lg object-cover bg-surface-container-highest border border-outline-variant" onerror="this.src='/favicon.png';this.onerror=null;" alt="Avatar">
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-1.5">
              <p class="text-sm font-semibold text-on-surface truncate">${alias || sid}</p>
              ${alias ? `<span class="text-[10px] text-on-surface-variant font-mono">(${sid})</span>` : ''}
            </div>
            <p class="text-xs text-on-surface-variant mt-0.5">${visits} recorded visit${visits > 1 ? 's' : ''}</p>
            <p class="text-[11px] text-error font-medium mt-0.5">⚠️ Will be merged &amp; removed</p>
          </div>
        </div>
      `;
    }

    const targetSelect = modal.querySelector('#merge-target-select');
    const targetCard = modal.querySelector('#merge-target-card');

    function updateTargetCard() {
      if (!targetCard) return;
      const targetId = targetSelect ? targetSelect.value : null;
      if (!targetId) {
        targetCard.innerHTML = '';
        targetCard.classList.add('hidden');
        return;
      }
      const targetVisitor = visitors.find((v) => v.subject_id === targetId) || { subject_id: targetId };
      const tSid = esc(targetVisitor.subject_id);
      const tAlias = esc(targetVisitor.alias || '');
      const tAvatarUrl = targetVisitor.avatar_path || `/api/workplace/avatar/${tSid}`;
      const tVisits = targetVisitor.total_visits || 1;

      targetCard.innerHTML = `
        <div class="flex items-center gap-3 p-3 rounded-xl bg-surface-container border border-primary/40">
          <img src="${tAvatarUrl}" class="w-12 h-12 rounded-lg object-cover bg-surface-container-highest border border-primary/30" onerror="this.src='/favicon.png';this.onerror=null;" alt="Target Avatar">
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-1.5">
              <p class="text-sm font-semibold text-on-surface truncate">${tAlias || tSid}</p>
              ${tAlias ? `<span class="text-[10px] text-on-surface-variant font-mono">(${tSid})</span>` : ''}
            </div>
            <p class="text-xs text-on-surface-variant mt-0.5">${tVisits} recorded visit${tVisits > 1 ? 's' : ''}</p>
            <p class="text-[11px] text-primary font-medium mt-0.5">✅ Kept as Master Profile (receives merged data)</p>
          </div>
        </div>
      `;
      targetCard.classList.remove('hidden');
    }

    if (targetSelect) {
      const candidates = visitors.filter((v) => v.subject_id !== sid);
      if (candidates.length === 0) {
        targetSelect.innerHTML = `<option value="">No other customer profiles available to merge into</option>`;
      } else {
        targetSelect.innerHTML = `
          <option value="">Select target customer profile...</option>
          ${candidates
            .map((c) => {
              const cSid = esc(c.subject_id);
              const cName = c.alias ? `${esc(c.alias)} (${cSid})` : cSid;
              const cVis = c.total_visits || 1;
              return `<option value="${cSid}">${cName} — ${cVis} visit${cVis > 1 ? 's' : ''}</option>`;
            })
            .join('')}
        `;
        if (preselectedTargetId && candidates.some((c) => c.subject_id === preselectedTargetId)) {
          targetSelect.value = preselectedTargetId;
        } else if (candidates.length === 1) {
          targetSelect.value = candidates[0].subject_id;
        }
      }

      targetSelect.onchange = updateTargetCard;
      updateTargetCard();
    }

    const confirmBtn = modal.querySelector('#merge-btn-confirm');
    if (confirmBtn) {
      confirmBtn.onclick = () => {
        const targetId = targetSelect ? targetSelect.value : null;
        if (!targetId) {
          alert('Please select a target profile to merge into.');
          return;
        }
        if (confirm(`Are you sure you want to merge "${alias || sid}" into selected profile "${targetId}"?\n\nThis will combine visit logs, stay durations, and appearance Re-ID models so future visits match as one customer.`)) {
          executeMerge(sid, targetId);
        }
      };
    }

    modal.classList.remove('hidden');
  }

  function closeModals() {
    const snapModal = document.getElementById('visit-snapshot-modal');
    if (snapModal) snapModal.classList.add('hidden');
    const mergeModal = document.getElementById('visit-merge-modal');
    if (mergeModal) {
      mergeModal.classList.add('hidden');
      const targetCard = mergeModal.querySelector('#merge-target-card');
      if (targetCard) {
        targetCard.innerHTML = '';
        targetCard.classList.add('hidden');
      }
    }
  }

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModals();
  });

  function renderCalendarGrid(container) {
    const gridEl = container.querySelector('#calendar-month-grid');
    if (!gridEl) return;

    if (!calendarMonth) {
      calendarMonth = new Date().toISOString().substring(0, 7);
    }
    const [yStr, mStr] = calendarMonth.split('-');
    const year = Number(yStr);
    const month = Number(mStr) - 1;

    const firstDayOfWeek = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const daysInPrevMonth = new Date(year, month, 0).getDate();

    const mapByDate = {};
    (cachedCalendar || []).forEach((item) => {
      if (item && item.date) mapByDate[item.date] = item;
    });

    let cellsHtml = '';

    // Leading days from previous month
    for (let i = firstDayOfWeek - 1; i >= 0; i--) {
      const dayNum = daysInPrevMonth - i;
      cellsHtml += `
        <div class="h-14 p-1 rounded-lg bg-surface-container-lowest/30 border border-outline-variant/20 text-on-surface-variant/30 flex flex-col justify-between select-none">
          <span class="text-[11px] font-mono">${dayNum}</span>
        </div>
      `;
    }

    // Days of current month
    const todayStr = new Date().toISOString().substring(0, 10);

    for (let day = 1; day <= daysInMonth; day++) {
      const padM = String(month + 1).padStart(2, '0');
      const padD = String(day).padStart(2, '0');
      const dateKey = `${year}-${padM}-${padD}`;
      const dataItem = mapByDate[dateKey];
      const hasVisits = Boolean(dataItem && dataItem.total_visits > 0);
      const visitsCount = hasVisits ? dataItem.total_visits : 0;
      const uniqCount = hasVisits ? dataItem.unique_visitors : 0;
      const avgDwell = hasVisits ? formatDwell(dataItem.avg_dwell_seconds) : '';
      const isSelected = currentDateFilter === dateKey;
      const isToday = todayStr === dateKey;

      let badgeHtml = '';
      let cellBg = 'bg-surface-container border border-outline-variant/40 hover:border-primary/50';
      if (isSelected) {
        cellBg = 'bg-primary/15 border-2 border-primary shadow-sm';
      } else if (hasVisits) {
        if (visitsCount >= 10) {
          badgeHtml = `<span class="px-1.5 py-0.5 text-[10px] font-mono font-bold rounded-md bg-primary text-surface font-semibold">${visitsCount}</span>`;
        } else if (visitsCount >= 4) {
          badgeHtml = `<span class="px-1.5 py-0.5 text-[10px] font-mono font-semibold rounded-md bg-primary/30 text-primary border border-primary/40">${visitsCount}</span>`;
        } else {
          badgeHtml = `<span class="px-1.5 py-0.5 text-[10px] font-mono rounded-md bg-surface-container-highest text-primary">${visitsCount}</span>`;
        }
      }

      cellsHtml += `
        <button
          type="button"
          class="h-14 p-1.5 rounded-lg ${cellBg} flex flex-col justify-between transition-all cursor-pointer text-left group relative"
          data-calendar-date="${dateKey}"
          title="${dateKey}: ${visitsCount} visits${hasVisits ? ` (${uniqCount} unique, avg dwell ${avgDwell})` : ''}"
        >
          <div class="flex items-center justify-between w-full">
            <span class="text-xs font-mono font-medium ${isToday ? 'text-primary font-bold' : 'text-on-surface'}">${day}</span>
            ${isToday ? `<span class="w-1.5 h-1.5 rounded-full bg-primary" title="Today"></span>` : ''}
          </div>
          <div class="flex items-center justify-between w-full mt-1">
            ${badgeHtml}
            ${hasVisits && !badgeHtml ? `<span class="w-2 h-2 rounded-full bg-primary/60"></span>` : ''}
          </div>
        </button>
      `;
    }

    gridEl.innerHTML = cellsHtml;

    gridEl.querySelectorAll('[data-calendar-date]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const d = btn.getAttribute('data-calendar-date');
        if (currentDateFilter === d) {
          currentDateFilter = null;
        } else {
          currentDateFilter = d;
        }
        render(container);
      });
    });
  }

  function render(container) {
    if (!container) return;
    targetContainer = container;
    startTimers();

    const alreadyMounted = Boolean(container.querySelector('#table-visitors-tbody'));

    if (!cachedData && !alreadyMounted) {
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

    Promise.all([fetchDetailedReport(), fetchCalendarSummary(calendarMonth)])
      .then(([data, cal]) => {
        cachedData = data;
        cachedCalendar = cal;
        renderDOM(container, data, alreadyMounted);
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

  function renderDOM(container, data, isPatchOnly) {
    const summary = data.summary || {};
    const visitors = data.visitors || [];
    const openSessions = data.open_sessions || [];

    let filteredVisitors = visitors.filter((v) => {
      if (!searchQuery) return true;
      const q = searchQuery.toLowerCase();
      const sid = (v.subject_id || '').toLowerCase();
      const alias = (v.alias || '').toLowerCase();
      return sid.includes(q) || alias.includes(q);
    });

    if (currentSort === 'recent') {
      filteredVisitors.sort((a, b) => (b.last_seen_at || '').localeCompare(a.last_seen_at || ''));
    } else if (currentSort === 'frequency') {
      filteredVisitors.sort((a, b) => (b.total_visits || 0) - (a.total_visits || 0));
    } else if (currentSort === 'dwell') {
      filteredVisitors.sort((a, b) => (b.avg_dwell_seconds || 0) - (a.avg_dwell_seconds || 0));
    }

    const existingTable = container.querySelector('#table-visitors-tbody');
    const existingSearch = container.querySelector('#inp-visits-search');
    const isUserInteracting = existingSearch && (document.activeElement === existingSearch || existingSearch.value !== searchQuery);

    if (isPatchOnly && existingTable && !isUserInteracting) {
      const kpiToday = container.querySelector('#kpi-today-val');
      if (kpiToday) kpiToday.textContent = summary.today_unique || 0;
      const kpiTodaySub = container.querySelector('#kpi-today-sub');
      if (kpiTodaySub) kpiTodaySub.textContent = `${summary.today_visits || 0} visits total`;

      const kpiWeek = container.querySelector('#kpi-week-val');
      if (kpiWeek) kpiWeek.textContent = summary.week_unique || 0;
      const kpiWeekSub = container.querySelector('#kpi-week-sub');
      if (kpiWeekSub) kpiWeekSub.textContent = `${summary.week_visits || 0} visits past week`;

      const kpiDwell = container.querySelector('#kpi-dwell-val');
      if (kpiDwell) kpiDwell.textContent = formatDwell(summary.avg_dwell_seconds);

      const kpiReturn = container.querySelector('#kpi-return-val');
      if (kpiReturn) kpiReturn.textContent = `${summary.returning_rate || 0}%`;

      const kpiActive = container.querySelector('#kpi-active-val');
      if (kpiActive) kpiActive.textContent = openSessions.length;

      const sessionsWrap = container.querySelector('#wrap-active-sessions');
      if (sessionsWrap) {
        sessionsWrap.innerHTML = renderActiveSessionsHtml(openSessions);
        attachSessionCardEvents(sessionsWrap);
      }

      existingTable.innerHTML = renderVisitorRowsHtml(filteredVisitors);
      attachRowEvents(container);
      return;
    }

    if (!calendarMonth) {
      calendarMonth = new Date().toISOString().substring(0, 7);
    }

    let html = `
      <div class="space-y-5">
        <!-- Controls & Filter Header -->
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-surface-container border border-outline-variant">
          <div>
            <div class="flex items-center gap-2">
              <span class="inline-block w-2.5 h-2.5 rounded-full bg-primary animate-pulse"></span>
              <h2 class="text-base font-semibold text-on-surface">Customer Attendance &amp; Visit Analytics</h2>
              <span class="px-2 py-0.5 text-[11px] font-mono rounded bg-primary/10 text-primary border border-primary/20">Venue Mode</span>
            </div>
            <p class="text-xs text-on-surface-variant mt-1">Real-time attendance, live dwell timers, visit calendar density, and customer profile merging.</p>
          </div>
          <div class="flex items-center gap-2 flex-wrap">
            <button id="btn-toggle-calendar" class="px-3 py-1.5 text-xs font-medium rounded-lg ${showCalendar ? 'bg-primary text-surface font-semibold' : 'bg-surface-container-high border border-outline-variant text-on-surface hover:bg-surface-container-highest'} transition-colors flex items-center gap-1.5">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
              <span>${showCalendar ? 'Hide Calendar' : 'Visit Calendar'}</span>
              ${currentDateFilter ? `<span class="w-1.5 h-1.5 rounded-full bg-primary"></span>` : ''}
            </button>
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

        <!-- Expandable Calendar Section -->
        <div id="section-calendar" class="${showCalendar ? '' : 'hidden'} rounded-xl border border-outline-variant bg-surface-container-low p-4 space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-outline-variant/60 pb-3">
            <div class="flex items-center gap-3">
              <button id="cal-prev-month" class="p-1.5 rounded-lg border border-outline-variant hover:bg-surface-container text-on-surface transition-colors">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
              </button>
              <h3 id="cal-month-title" class="text-sm font-semibold text-on-surface font-mono">${getMonthYearHeader(calendarMonth)}</h3>
              <button id="cal-next-month" class="p-1.5 rounded-lg border border-outline-variant hover:bg-surface-container text-on-surface transition-colors">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
              </button>
              <button id="cal-today-btn" class="px-2 py-1 text-[11px] rounded font-medium bg-surface-container border border-outline-variant text-on-surface hover:border-primary transition-colors">
                Today
              </button>
            </div>
            <div class="flex items-center gap-3 text-xs text-on-surface-variant flex-wrap">
              <span class="flex items-center gap-1"><span class="w-2 h-2 rounded bg-surface-container-highest border border-outline-variant"></span> 1–3 visits</span>
              <span class="flex items-center gap-1"><span class="w-2 h-2 rounded bg-primary/40"></span> 4–9 visits</span>
              <span class="flex items-center gap-1"><span class="w-2 h-2 rounded bg-primary"></span> 10+ peak</span>
              ${
                currentDateFilter
                  ? `<button id="cal-btn-clear-date" class="ml-2 px-2 py-0.5 text-xs rounded bg-error/10 text-error border border-error/30 hover:bg-error/20 flex items-center gap-1">✕ Clear Date Filter (${formatDateShort(currentDateFilter)})</button>`
                  : ''
              }
            </div>
          </div>

          <div class="grid grid-cols-7 gap-1 text-center text-[11px] font-mono text-on-surface-variant/80 font-medium">
            <div>Sun</div><div>Mon</div><div>Tue</div><div>Wed</div><div>Thu</div><div>Fri</div><div>Sat</div>
          </div>

          <div id="calendar-month-grid" class="grid grid-cols-7 gap-1"></div>
        </div>

        <!-- 5-Card KPI Grid -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div class="rounded-xl border border-outline-variant bg-surface-container p-3">
            <div class="flex items-center justify-between">
              <p class="text-xs text-on-surface-variant">Today Unique</p>
              <span class="text-[10px] font-mono text-primary bg-primary/10 px-1.5 py-0.5 rounded">Active</span>
            </div>
            <p id="kpi-today-val" class="font-mono-data text-xl text-primary font-bold mt-1">${summary.today_unique || 0}</p>
            <p id="kpi-today-sub" class="text-[11px] text-on-surface-variant mt-0.5">${summary.today_visits || 0} visits total</p>
          </div>

          <div class="rounded-xl border border-outline-variant bg-surface-container p-3">
            <p class="text-xs text-on-surface-variant">7-Day Unique</p>
            <p id="kpi-week-val" class="font-mono-data text-xl text-on-surface font-bold mt-1">${summary.week_unique || 0}</p>
            <p id="kpi-week-sub" class="text-[11px] text-on-surface-variant mt-0.5">${summary.week_visits || 0} visits past week</p>
          </div>

          <div class="rounded-xl border border-outline-variant bg-surface-container p-3">
            <p class="text-xs text-on-surface-variant">Average Stay</p>
            <p id="kpi-dwell-val" class="font-mono-data text-xl text-on-surface font-bold mt-1">${formatDwell(summary.avg_dwell_seconds)}</p>
            <p class="text-[11px] text-on-surface-variant mt-0.5">Per completed visit</p>
          </div>

          <div class="rounded-xl border border-outline-variant bg-surface-container p-3">
            <p class="text-xs text-on-surface-variant">Returning Rate</p>
            <p id="kpi-return-val" class="font-mono-data text-xl text-on-surface font-bold mt-1">${summary.returning_rate || 0}%</p>
            <p class="text-[11px] text-on-surface-variant mt-0.5">Customers with &gt;1 visit</p>
          </div>

          <div class="rounded-xl border border-primary/30 bg-primary/5 p-3 col-span-2 md:col-span-1">
            <div class="flex items-center gap-1.5">
              <span class="w-2 h-2 rounded-full bg-primary animate-ping"></span>
              <p class="text-xs font-medium text-primary">In Venue Now</p>
            </div>
            <p id="kpi-active-val" class="font-mono-data text-xl text-primary font-bold mt-1">${openSessions.length}</p>
            <p class="text-[11px] text-primary/80 mt-0.5">Active inside zones</p>
          </div>
        </div>

        <!-- Active In-Venue Visitors Cards -->
        <div id="wrap-active-sessions">
          ${renderActiveSessionsHtml(openSessions)}
        </div>

        <!-- Directory & Filter Bar -->
        <div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div class="flex items-center gap-2">
              <h3 class="text-sm font-semibold text-on-surface">Customer Profiles &amp; Visit History</h3>
              ${
                currentDateFilter
                  ? `<span class="px-2 py-0.5 text-xs font-medium rounded-full bg-primary/20 text-primary border border-primary/30 flex items-center gap-1">📅 ${formatDateShort(currentDateFilter)} <button id="btn-clear-date-badge" class="hover:opacity-75 font-bold" title="Clear date filter">✕</button></span>`
                  : ''
              }
            </div>
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
              <p class="text-sm">No visitor records found${currentDateFilter ? ` for ${formatDateShort(currentDateFilter)}` : ''}.</p>
              <p class="text-xs mt-1">Visitors will automatically appear here once detected on camera.</p>
              ${
                currentDateFilter
                  ? `<button id="btn-clear-empty-date" class="mt-3 px-3 py-1 text-xs rounded bg-surface-container border border-outline-variant text-primary hover:border-primary transition-colors">Show All Dates</button>`
                  : ''
              }
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
                    <th class="py-2.5 px-3 font-medium">First &amp; Last Seen</th>
                    <th class="py-2.5 px-3 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody id="table-visitors-tbody" class="divide-y divide-outline-variant/60">
                  ${renderVisitorRowsHtml(filteredVisitors)}
                </tbody>
              </table>
            </div>
          `
          }
        </div>
      </div>

      <!-- Snapshot Zoom Modal -->
      <div id="visit-snapshot-modal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 hidden">
        <div class="bg-surface-container-high border border-outline-variant rounded-2xl max-w-md w-full overflow-hidden shadow-2xl animate-in fade-in zoom-in-95">
          <div class="p-4 border-b border-outline-variant flex items-center justify-between">
            <div>
              <h3 id="snapshot-modal-title" class="text-sm font-semibold text-on-surface">Customer Snapshot</h3>
              <p id="snapshot-modal-sub" class="text-xs text-on-surface-variant font-mono">Visitor ID</p>
            </div>
            <button id="snapshot-modal-btn-close" class="p-1 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
          </div>
          <div class="p-4 space-y-4">
            <div class="w-full h-64 rounded-xl overflow-hidden bg-black/50 border border-outline-variant flex items-center justify-center relative group">
              <img id="snapshot-modal-img" src="/favicon.png" class="w-full h-full object-contain" alt="Enlarged Avatar">
            </div>
            <div class="flex items-center justify-between">
              <div id="snapshot-modal-status"></div>
              <p id="snapshot-modal-visits" class="text-xs font-mono font-medium text-primary"></p>
            </div>
            <div class="grid grid-cols-2 gap-2 text-xs bg-surface-container p-3 rounded-xl border border-outline-variant/60 font-mono">
              <div>
                <p class="text-[10px] text-on-surface-variant">First Seen</p>
                <p id="snapshot-modal-first-seen" class="text-on-surface font-medium mt-0.5 truncate">—</p>
              </div>
              <div>
                <p class="text-[10px] text-on-surface-variant">Last Seen</p>
                <p id="snapshot-modal-last-seen" class="text-on-surface font-medium mt-0.5 truncate">—</p>
              </div>
              <div class="col-span-2 pt-2 border-t border-outline-variant/40">
                <p class="text-[10px] text-on-surface-variant">Dwell Duration</p>
                <p id="snapshot-modal-dwell" class="text-on-surface font-medium mt-0.5 truncate">—</p>
              </div>
            </div>
          </div>
          <div class="p-3 bg-surface-container border-t border-outline-variant flex items-center justify-between gap-2">
            <div class="flex items-center gap-2">
              <button id="snapshot-modal-btn-edit" class="px-3 py-1.5 text-xs rounded-lg border border-outline-variant hover:border-primary text-on-surface hover:text-primary transition-colors">
                Edit Name
              </button>
              <button id="snapshot-modal-btn-merge" class="px-3 py-1.5 text-xs rounded-lg border border-outline-variant hover:border-primary text-on-surface hover:text-primary transition-colors flex items-center gap-1">
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
                Merge
              </button>
            </div>
            <button id="snapshot-modal-btn-close-btm" class="px-4 py-1.5 text-xs rounded-lg bg-surface-container-high border border-outline-variant text-on-surface hover:bg-surface-container-highest transition-colors">
              Close
            </button>
          </div>
        </div>
      </div>

      <!-- Merge Profiles Modal -->
      <div id="visit-merge-modal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 hidden">
        <div class="bg-surface-container-high border border-outline-variant rounded-2xl max-w-md w-full max-h-[90vh] flex flex-col overflow-hidden shadow-2xl animate-in fade-in zoom-in-95">
          <div class="p-4 border-b border-outline-variant flex items-center justify-between shrink-0">
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full bg-primary"></span>
              <h3 class="text-sm font-semibold text-on-surface">Merge Visitor Profiles</h3>
            </div>
            <button id="merge-modal-btn-close" class="p-1 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container transition-colors">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
          </div>
          <div class="p-4 space-y-4 overflow-y-auto flex-1">
            <p class="text-xs text-on-surface-variant leading-relaxed">
              Use this when a customer was enrolled under a second profile because they changed clothing, hair, or appearance. Combining them merges all visit history, dwell duration, and Re-ID models into one profile.
            </p>

            <div>
              <p class="text-xs font-medium text-on-surface mb-1.5">1. Source Profile to Merge:</p>
              <div id="merge-source-card"></div>
            </div>

            <div class="flex justify-center text-on-surface-variant">
              <svg class="w-5 h-5 text-primary animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>
            </div>

            <div>
              <label for="merge-target-select" class="block text-xs font-medium text-on-surface mb-1.5">2. Target Profile (Keep as Master):</label>
              <select id="merge-target-select" class="w-full px-3 py-2 text-xs rounded-xl bg-surface-container border border-outline-variant text-on-surface focus:outline-none focus:border-primary">
              </select>
              <div id="merge-target-card" class="mt-2 hidden"></div>
            </div>

            <div class="p-3 rounded-xl bg-surface-container border border-primary/20 text-[11px] text-on-surface-variant space-y-1">
              <p class="font-semibold text-primary">✨ What happens after merge:</p>
              <p>• All past visits and durations are preserved under the master profile.</p>
              <p>• Appearance Re-ID embeddings are combined so the camera recognizes both outfits in the future.</p>
              <p>• The source profile ID is cleanly deleted.</p>
            </div>
          </div>
          <div class="p-3 bg-surface-container border-t border-outline-variant flex items-center justify-end gap-2 shrink-0">
            <button id="merge-modal-btn-cancel" class="px-3 py-1.5 text-xs rounded-lg border border-outline-variant text-on-surface hover:bg-surface-container-high transition-colors">
              Cancel
            </button>
            <button id="merge-btn-confirm" class="px-4 py-1.5 text-xs font-semibold rounded-lg bg-primary text-surface hover:opacity-90 transition-opacity flex items-center gap-1.5">
              Confirm &amp; Combine Data
            </button>
          </div>
        </div>
      </div>
    `;

    container.innerHTML = html;

    if (showCalendar) {
      renderCalendarGrid(container);
    }

    const resetBtn = container.querySelector('#btn-reset-all-visits');
    if (resetBtn) resetBtn.addEventListener('click', resetAllVisits);

    const refreshBtn = container.querySelector('#btn-refresh-visits');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        refreshBtn.classList.add('opacity-50');
        render(container);
      });
    }

    const toggleCalBtn = container.querySelector('#btn-toggle-calendar');
    if (toggleCalBtn) {
      toggleCalBtn.addEventListener('click', () => {
        showCalendar = !showCalendar;
        const calSection = container.querySelector('#section-calendar');
        if (calSection) {
          calSection.classList.toggle('hidden', !showCalendar);
          if (showCalendar) renderCalendarGrid(container);
        }
        toggleCalBtn.classList.toggle('bg-primary', showCalendar);
        toggleCalBtn.classList.toggle('text-surface', showCalendar);
        toggleCalBtn.classList.toggle('font-semibold', showCalendar);
      });
    }

    const clearDateBtn = container.querySelector('#cal-btn-clear-date');
    if (clearDateBtn) {
      clearDateBtn.addEventListener('click', () => {
        currentDateFilter = null;
        render(container);
      });
    }

    const clearBadgeBtn = container.querySelector('#btn-clear-date-badge');
    if (clearBadgeBtn) {
      clearBadgeBtn.addEventListener('click', () => {
        currentDateFilter = null;
        render(container);
      });
    }

    const clearEmptyDateBtn = container.querySelector('#btn-clear-empty-date');
    if (clearEmptyDateBtn) {
      clearEmptyDateBtn.addEventListener('click', () => {
        currentDateFilter = null;
        render(container);
      });
    }

    const calPrevMonth = container.querySelector('#cal-prev-month');
    if (calPrevMonth) {
      calPrevMonth.addEventListener('click', () => {
        const [yStr, mStr] = calendarMonth.split('-');
        let y = Number(yStr);
        let m = Number(mStr) - 1;
        if (m < 1) { m = 12; y -= 1; }
        calendarMonth = `${y}-${String(m).padStart(2, '0')}`;
        fetchCalendarSummary(calendarMonth).then(() => {
          const t = container.querySelector('#cal-month-title');
          if (t) t.textContent = getMonthYearHeader(calendarMonth);
          renderCalendarGrid(container);
        });
      });
    }

    const calNextMonth = container.querySelector('#cal-next-month');
    if (calNextMonth) {
      calNextMonth.addEventListener('click', () => {
        const [yStr, mStr] = calendarMonth.split('-');
        let y = Number(yStr);
        let m = Number(mStr) + 1;
        if (m > 12) { m = 1; y += 1; }
        calendarMonth = `${y}-${String(m).padStart(2, '0')}`;
        fetchCalendarSummary(calendarMonth).then(() => {
          const t = container.querySelector('#cal-month-title');
          if (t) t.textContent = getMonthYearHeader(calendarMonth);
          renderCalendarGrid(container);
        });
      });
    }

    const calTodayBtn = container.querySelector('#cal-today-btn');
    if (calTodayBtn) {
      calTodayBtn.addEventListener('click', () => {
        calendarMonth = new Date().toISOString().substring(0, 7);
        fetchCalendarSummary(calendarMonth).then(() => {
          const t = container.querySelector('#cal-month-title');
          if (t) t.textContent = getMonthYearHeader(calendarMonth);
          renderCalendarGrid(container);
        });
      });
    }

    const searchInput = container.querySelector('#inp-visits-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        const tbody = container.querySelector('#table-visitors-tbody');
        if (tbody && cachedData) {
          const vList = (cachedData.visitors || []).filter((v) => {
            if (!searchQuery) return true;
            const q = searchQuery.toLowerCase();
            return (v.subject_id || '').toLowerCase().includes(q) || (v.alias || '').toLowerCase().includes(q);
          });
          tbody.innerHTML = renderVisitorRowsHtml(vList);
          attachRowEvents(container);
        }
      });
    }

    const sortSelect = container.querySelector('#sel-visits-sort');
    if (sortSelect) {
      sortSelect.addEventListener('change', (e) => {
        currentSort = e.target.value;
        const tbody = container.querySelector('#table-visitors-tbody');
        if (tbody && cachedData) {
          renderDOM(container, cachedData, true);
        }
      });
    }

    const snapClose = container.querySelector('#snapshot-modal-btn-close');
    if (snapClose) snapClose.addEventListener('click', closeModals);
    const snapCloseBtm = container.querySelector('#snapshot-modal-btn-close-btm');
    if (snapCloseBtm) snapCloseBtm.addEventListener('click', closeModals);
    const snapModal = container.querySelector('#visit-snapshot-modal');
    if (snapModal) {
      snapModal.addEventListener('click', (e) => {
        if (e.target === snapModal) closeModals();
      });
    }

    const mergeClose = container.querySelector('#merge-modal-btn-close');
    if (mergeClose) mergeClose.addEventListener('click', closeModals);
    const mergeCancel = container.querySelector('#merge-modal-btn-cancel');
    if (mergeCancel) mergeCancel.addEventListener('click', closeModals);
    const mergeModal = container.querySelector('#visit-merge-modal');
    if (mergeModal) {
      mergeModal.addEventListener('click', (e) => {
        if (e.target === mergeModal) closeModals();
      });
    }

    attachSessionCardEvents(container);
    attachRowEvents(container);
  }

  function renderActiveSessionsHtml(openSessions) {
    if (!openSessions || openSessions.length === 0) return '';
    return `
      <div class="rounded-xl border border-primary/20 bg-surface-container-low p-4">
        <h3 class="text-sm font-semibold text-on-surface mb-3 flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-primary animate-ping"></span>
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
                <div class="rounded-lg border border-outline-variant bg-surface-container p-3 flex items-center gap-3 hover:border-primary/40 transition-colors">
                  <div class="relative group cursor-pointer" data-zoom-id="${sid}" title="Click to view full photo">
                    <img src="${avatarUrl}" class="w-12 h-12 rounded-lg object-cover bg-surface-container-highest border border-outline-variant group-hover:ring-2 group-hover:ring-primary transition-all" onerror="this.src='/favicon.png';this.onerror=null;" alt="Avatar">
                    <span class="absolute inset-0 bg-black/40 rounded-lg opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-primary">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"></path></svg>
                    </span>
                  </div>
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center justify-between">
                      <p class="text-sm font-medium text-on-surface truncate">${sid}</p>
                      <span class="px-1.5 py-0.5 text-[10px] font-mono rounded bg-primary/20 text-primary">Active</span>
                    </div>
                    <p class="text-xs text-on-surface-variant truncate">In ${zname}</p>
                    <p class="font-mono-data text-xs text-primary font-semibold mt-0.5">
                      <span data-live-started="${session.started_at}">⏱️ ${dwell}</span>
                      <span class="text-[10px] text-on-surface-variant font-normal">(since ${started})</span>
                    </p>
                  </div>
                </div>
              `;
            })
            .join('')}
        </div>
      </div>
    `;
  }

  function renderVisitorRowsHtml(filteredVisitors) {
    return filteredVisitors
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
                <div class="relative group cursor-pointer" data-zoom-id="${sid}" title="Click to view full photo">
                  <img src="${avatarUrl}" class="w-9 h-9 rounded-lg object-cover bg-surface-container-highest border border-outline-variant group-hover:ring-2 group-hover:ring-primary transition-all" onerror="this.src='/favicon.png';this.onerror=null;" alt="Avatar">
                  <span class="absolute inset-0 bg-black/40 rounded-lg opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-primary">
                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"></path></svg>
                  </span>
                </div>
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
                  ? `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-primary/10 text-primary border border-primary/20"><span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>In Venue</span>`
                  : `<span class="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-surface-container-highest text-on-surface-variant">Departed</span>`
              }
            </td>
            <td class="py-2.5 px-3">
              <p class="font-mono-data font-semibold text-on-surface">${totalVisits} <span class="font-normal text-[11px] text-on-surface-variant">visit${totalVisits === 1 ? '' : 's'}</span></p>
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
              <div class="flex items-center justify-end gap-1.5">
                <button class="btn-merge-profile px-2 py-1 text-xs rounded border border-outline-variant hover:border-primary text-on-surface-variant hover:text-primary transition-colors flex items-center gap-1" data-id="${sid}" title="Merge duplicate customer profile">
                  <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
                  Merge
                </button>
                <button class="btn-edit-alias px-2.5 py-1 text-xs rounded border border-outline-variant hover:border-primary text-on-surface hover:text-primary transition-colors" data-id="${sid}" data-alias="${alias}">
                  ${alias ? 'Edit Name' : '+ Set Name'}
                </button>
              </div>
            </td>
          </tr>
        `;
      })
      .join('');
  }

  function attachSessionCardEvents(container) {
    const zoomTriggers = container.querySelectorAll('[data-zoom-id]');
    zoomTriggers.forEach((el) => {
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = el.getAttribute('data-zoom-id');
        if (id) openSnapshotModal(id);
      });
    });
  }

  function attachRowEvents(container) {
    const editBtns = container.querySelectorAll('.btn-edit-alias');
    editBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        const currentA = btn.getAttribute('data-alias');
        editAlias(id, currentA);
      });
    });

    const mergeBtns = container.querySelectorAll('.btn-merge-profile');
    mergeBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        if (id) openMergeModal(id);
      });
    });

    const zoomTriggers = container.querySelectorAll('[data-zoom-id]');
    zoomTriggers.forEach((el) => {
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = el.getAttribute('data-zoom-id');
        if (id) openSnapshotModal(id);
      });
    });
  }

  window.VisitsReportingComponent = {
    render: render,
    pollNow: pollUpdates,
    openSnapshot: openSnapshotModal,
    openMerge: openMergeModal,
  };
})();
