/* Order Block page shell: relays "new minute" events and the stock search's keys to the
   terminal frame and shows the pre-open / connection states around it. The terminal itself
   is 02.html. */
(function () {
  'use strict';
  const frame = OB.$('#obFrame'), wait = OB.$('#obWait'), status = OB.$('#obStatus');
  const pills = OB.$$('[data-live-pill]'); // the menu's copy and the phone bar's copy
  let countdown = null, lastOk = Date.now(), live = false;
  let empty = false, open = '09:15';  // the frame's last state: no rankings yet / the session's open
  let mkt = null;                     // the market status (/api/market/status, then 'market' events)
  let conn = '', feed = '';           // the status line: the frame's connection, the server's feed note

  const refresh = () => { try { frame.contentWindow.postMessage({ t: 'ob-refresh' }, location.origin); } catch (e) { } };
  // spread the per-minute refetch over ~4s so thousands of open terminals don't hit the server at once
  OB.on('ob', () => setTimeout(refresh, Math.random() * 4000));
  // the terminal repaints in the new colours when the sun / moon toggle (or the system theme) changes
  window.addEventListener('themechange', e => { try { frame.contentWindow.postMessage({ t: 'ob-theme', theme: e.detail }, location.origin); } catch (_) { } });
  // the server says when the session's data is late (feed: syncing | delayed) and when that ends
  OB.on('market', st => applyStatus(st));
  // the terminal's stock search opens from anywhere on the page: "/", Ctrl K (⌘K) and Ctrl F go to the frame, which
  // opens it (once the frame has the focus it handles them itself)
  document.addEventListener('keydown', e => {
    if (empty || e.defaultPrevented || e.isComposing) return;
    const t = e.target; if (t && t.closest && t.closest('input,textarea,select,[contenteditable]')) return;
    const k = (e.key || '').toLowerCase(), mod = (e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey;
    if (!((mod && (k === 'k' || k === 'f')) || (k === '/' && !e.metaKey && !e.ctrlKey && !e.altKey))) return;
    e.preventDefault();
    try { frame.focus(); frame.contentWindow.postMessage({ t: 'ob-find' }, location.origin); } catch (_) { }
  });

  // What the wait card says. Before the open: the countdown. During the session there is no data
  // only when something is wrong, so say what: today's minutes are being rebuilt (syncing), or no
  // live data is arriving (delayed; also assumed when the list is still empty three minutes in).
  function waitCopy(st) {
    if (st && st.state === 'open') {
      const late = st.opens_at && Date.now() / 1000 - st.opens_at > 180;
      if (st.feed === 'syncing') return ['Loading today’s session', 'Catching up on every minute since ' + open + '. Rankings appear here in a minute or two.'];
      if (st.feed === 'delayed' || late) return ['Live data is delayed', 'The exchange feed isn’t reaching us right now. Rankings appear here as soon as it’s back, no need to refresh.'];
      return ['Rankings are on their way', 'The first rankings land a minute after the open and refresh every minute.'];
    }
    return ['Session starts at ' + open, 'Yesterday’s data has been cleared. Rankings appear a minute after the open and refresh every minute.'];
  }

  function showWait() {
    const [title, text] = waitCopy(mkt);
    OB.$('#obWaitTitle').textContent = title;
    OB.$('#obWaitText').textContent = text;
    wait.hidden = false;
    clearInterval(countdown);
    const el = OB.$('#obCount');
    const opensAt = mkt && mkt.state !== 'open' ? mkt.opens_at : 0;
    if (!opensAt) { el.textContent = ''; return; }
    const tick = () => {
      const s = Math.max(0, Math.round(opensAt - Date.now() / 1000));
      if (s <= 0) { el.textContent = ''; clearInterval(countdown); return; }
      const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), ss = s % 60;
      el.textContent = (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + String(ss).padStart(2, '0');
    };
    tick(); countdown = setInterval(tick, 1000);
  }

  function applyStatus(st) {
    if (!st || !st.state) return;
    mkt = st;
    const note = st.state === 'open' ? st.feed : '';
    feed = note === 'delayed' ? 'Live data delayed, reconnecting to the exchange feed…' : note === 'syncing' ? 'Catching up with today’s session…' : '';
    if (empty) showWait();
    paintStatus();
  }

  async function loadStatus() {
    try { applyStatus(await OB.get(document.body.dataset.mktSrc || '/api/market/status')); } catch (_) { }
  }

  window.addEventListener('message', async e => {
    if (e.origin !== location.origin || !e.data) return;
    const d = e.data;
    if (d.t === 'ob-auth') { location.href = d.s === 402 ? '/checkout' : '/login?next=/app/orderblock'; return; }
    if (d.t === 'ob-error') { if (Date.now() - lastOk > 150000) { conn = 'Reconnecting to live data…'; paintStatus(); } return; }
    if (d.t !== 'ob-state') return;
    lastOk = Date.now(); conn = '';
    live = d.status === 'live';
    pills.forEach(p => { p.hidden = !live; });
    open = d.open || open;
    const wasEmpty = empty;
    empty = !!d.empty;
    if (empty) {
      if (!wasEmpty || !mkt) await loadStatus();
      showWait();
    } else {
      wait.hidden = true; clearInterval(countdown);
      if (!mkt) loadStatus(); // the status line (delayed / catching up) before the next 'market' event
    }
    paintStatus();
  });

  function paintStatus() {
    const t = conn || (empty ? '' : feed); // the wait card already explains an empty terminal
    status.hidden = !t;
    OB.$('#obStatusText').textContent = t;
  }
  // if the feed goes quiet during market hours, say so (the frame keeps retrying)
  setInterval(() => { if (live && Date.now() - lastOk > 180000) { conn = 'Waiting for the next update…'; paintStatus(); } }, 30000);
  // while the wait card is up, keep its reason current even if a 'market' event was missed
  setInterval(() => { if (empty && !document.hidden) loadStatus(); }, 30000);
})();
