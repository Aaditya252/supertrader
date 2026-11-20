/* OrderBlock charts — dependency-free SVG charts that follow the theme tokens.
   Every chart redraws on resize and on theme change.

   OBC.line(el, { labels, series:[{name, data, color:'brand'|'bull'|'#hex', area:true, dashed}], height, yFmt, xFmt, tipFmt, zero:true })
   OBC.bars(el, { labels, series:[{name, data, color}], stacked, height, yFmt, xFmt, right:{name, per, data, color, fmt, tipFmt} })
     right: an optional right-hand axis reading the same bars in a second unit (per of it for each 1 on the left, e.g.
       revenue at a fixed price per sale), on the same gridlines; data gives the tooltip's exact values
   OBC.hbars(el, { items:[{label, value, color, sub}], fmt, max })
   OBC.donut(el, { items:[{label, value, color}], fmt, center:{value, label}, size })
   OBC.spark(el, data, { color, height, area })
   OBC.calendar(el, { values:{'YYYY-MM-DD': number}, from:'YYYY-MM-DD', to:'YYYY-MM-DD', fmt, tip })
   OBC.matrix(el, { rows:[...], cols:[...], values:[[...]], fmt, color })
   OBC.funnel(el, { steps:[{label, value}], fmt })
*/
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const root = document.documentElement;
  const PALETTE = ['brand', 'sky', 'pink', 'amber', 'teal', 'violet', 'bull', 'bear'];
  const css = n => getComputedStyle(root).getPropertyValue('--' + n).trim();
  const col = c => !c ? css('brand') : (c[0] === '#' || c.startsWith('rgb')) ? c : css(c) || c;
  const s = (tag, a = {}) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const esc = x => String(x ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmtN = v => (window.OB && OB.fmt ? OB.fmt.compact(v) : String(v));
  // drawing width: the content box, so a padded container doesn't push the chart past its edge
  const innerW = el => { const cs = getComputedStyle(el); return el.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0); };

  const registry = new Set();
  const ro = 'ResizeObserver' in window ? new ResizeObserver(ents => ents.forEach(e => { const el = e.target; if (el._obc && Math.abs(el._obcW - el.clientWidth) > 2) draw(el); })) : null;
  window.addEventListener('themechange', () => registry.forEach(el => el.isConnected ? draw(el) : registry.delete(el)));

  function mount(el, type, cfg) {
    el._obc = { type, cfg }; registry.add(el);
    if (ro && !el._obcRO) { ro.observe(el); el._obcRO = true; }
    draw(el);
    return { update(c) { el._obc.cfg = Object.assign(el._obc.cfg, c); draw(el); }, el };
  }
  function draw(el) { hideTip(el); el._obcW = el.clientWidth; const { type, cfg } = el._obc; DRAW[type](el, cfg); }

  /* ---------- tooltip ----------
     One tip for every chart, on the viewport. showTip(html, e, off) is called from a mark's mousemove (e: that event,
     off: undoes the mark's hover look). The tip goes when the pointer leaves the mark, when anything scrolls, resizes
     or takes the focus (core.js floating layers), when a press lands anywhere but that mark (a phone keeps a tapped
     mark "hovered", so there the tip stays until the next tap or scroll), and when its chart redraws or is removed. */
  let tip, tipAt = null, tipOff = null;
  const gone = 'MutationObserver' in window ? new MutationObserver(() => { if (tipAt && !tipAt.isConnected) hideTip(); }) : null;
  function showTip(html, e, off) {
    if (!tip) {
      tip = document.createElement('div'); tip.className = 'obc-tip'; document.body.append(tip);
      if (window.OB && OB.float) OB.float(tip, hideTip);
    }
    if (tipOff && tipOff !== off) tipOff();
    if (!tipAt && gone) gone.observe(document.body, { childList: true, subtree: true });
    tipAt = e.currentTarget || e.target; tipOff = off || null;
    tip.innerHTML = html; tip.style.display = 'block';
    const w = tip.offsetWidth, h = tip.offsetHeight, x = e.clientX, y = e.clientY;
    let tx = x + 14, ty = y - h - 10;
    if (tx + w > innerWidth - 8) tx = x - w - 14;
    if (ty < 8) ty = y + 16;
    tx = Math.max(8, Math.min(tx, innerWidth - w - 8)); ty = Math.max(8, Math.min(ty, innerHeight - h - 8));
    tip.style.transform = `translate(${tx}px,${ty}px)`;
  }
  // within: only a tip that belongs to a mark inside this element (the mark the pointer left, a chart that redraws)
  function hideTip(within) {
    if (!tipAt || (within && within.nodeType === 1 && !within.contains(tipAt))) return;
    const off = tipOff;
    if (gone) gone.disconnect();
    tipAt = tipOff = null;
    if (off) off();
    tip.style.display = 'none';
  }
  document.addEventListener('pointerdown', e => { if (tipAt && !tipAt.contains(e.target)) hideTip(); }, true);

  function niceTicks(min, max, n = 4) {
    if (min === max) { max = min + 1; min = min - (min === 0 ? 0 : 1); }
    const span = max - min, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(x => span / x <= n) || mag * 10;
    const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
    const out = []; for (let v = lo; v <= hi + step / 2; v += step) out.push(+v.toFixed(10));
    return out;
  }

  const DRAW = {};

  /* ---------- line / area ---------- */
  DRAW.line = function (el, c) {
    const W = Math.max(innerW(el), 120), H = c.height || 220;
    const padL = c.yAxis === false ? 8 : 46, padR = 12, padT = 12, padB = c.xAxis === false ? 8 : 26;
    const labels = c.labels || [], n = labels.length;
    const series = (c.series || []).map((sr, i) => Object.assign({ color: PALETTE[i % PALETTE.length] }, sr));
    let lo = Infinity, hi = -Infinity;
    series.forEach(sr => sr.data.forEach(v => { if (v == null) return; lo = Math.min(lo, v); hi = Math.max(hi, v); }));
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (c.zero !== false) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    const ticks = niceTicks(lo, hi, c.ticks || 4); lo = ticks[0]; hi = ticks[ticks.length - 1];
    const x = i => padL + (n <= 1 ? (W - padL - padR) / 2 : i * (W - padL - padR) / (n - 1));
    const y = v => padT + (hi - v) / (hi - lo || 1) * (H - padT - padB);
    const svg = s('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: 'obc' });
    const grid = css('line'), ink3 = css('ink-3');
    if (c.yAxis !== false) ticks.forEach(t => {
      svg.append(s('line', { x1: padL, x2: W - padR, y1: y(t), y2: y(t), stroke: grid, 'stroke-dasharray': t === 0 ? '' : '2 4' }));
      const tx = s('text', { x: padL - 8, y: y(t) + 3.5, 'text-anchor': 'end', class: 'obc-ax' }); tx.textContent = (c.yFmt || fmtN)(t); svg.append(tx);
    });
    if (c.xAxis !== false && n) {
      const gapX = c.xGap || 64;
      const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor((W - padL) / gapX))));
      // the newest label always shows; a regular label too close to it gives way (they overlapped on phones)
      labels.forEach((l, i) => { const last = i === n - 1; if (!last && i % every) return; if (!last && i && x(n - 1) - x(i) < gapX) return; if (last && i && i % every && x(i) - x(0) < gapX) return; const tx = s('text', { x: x(i), y: H - 7, 'text-anchor': i === 0 && n > 1 ? 'start' : i === n - 1 && n > 1 ? 'end' : 'middle', class: 'obc-ax' }); tx.textContent = (c.xFmt || String)(l); svg.append(tx); });
    }
    const defs = s('defs'); svg.append(defs);
    series.forEach((sr, si) => {
      const color = col(sr.color);
      const pts = []; sr.data.forEach((v, i) => { if (v != null) pts.push([x(i), y(v)]); });
      if (!pts.length) return;
      const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('');
      if (sr.area !== false && (c.area || sr.area)) {
        const gid = 'g' + Math.random().toString(36).slice(2, 8);
        const g = s('linearGradient', { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 });
        g.append(s('stop', { offset: '0%', 'stop-color': color, 'stop-opacity': sr.fillOpacity ?? .22 }), s('stop', { offset: '100%', 'stop-color': color, 'stop-opacity': 0 }));
        defs.append(g);
        const base = y(Math.max(lo, Math.min(0, hi)));
        svg.append(s('path', { d: d + `L${pts[pts.length - 1][0]},${base}L${pts[0][0]},${base}Z`, fill: `url(#${gid})` }));
      }
      svg.append(s('path', { d, fill: 'none', stroke: color, 'stroke-width': sr.width || 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': sr.dashed ? '4 4' : '' }));
      if (pts.length === 1 || c.dots) pts.forEach(p => svg.append(s('circle', { cx: p[0], cy: p[1], r: 3, fill: color })));
    });
    // hover
    const cross = s('line', { y1: padT, y2: H - padB, stroke: ink3, 'stroke-dasharray': '3 3', opacity: 0 });
    const dots = series.map(sr => { const d = s('circle', { r: 4, fill: css('surface'), stroke: col(sr.color), 'stroke-width': 2, opacity: 0 }); return d; });
    svg.append(cross, ...dots);
    const hit = s('rect', { x: padL, y: 0, width: Math.max(0, W - padL - padR), height: H, fill: 'transparent' });
    svg.append(hit);
    const off = () => { cross.setAttribute('opacity', 0); dots.forEach(d => d.setAttribute('opacity', 0)); };
    hit.addEventListener('mousemove', e => {
      const r = svg.getBoundingClientRect(); const mx = e.clientX - r.left;
      const i = Math.max(0, Math.min(n - 1, Math.round((mx - padL) / ((W - padL - padR) / Math.max(1, n - 1)))));
      cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('opacity', 1);
      let rows = '';
      series.forEach((sr, si) => {
        const v = sr.data[i]; const d = dots[si];
        if (v == null) { d.setAttribute('opacity', 0); return; }
        d.setAttribute('cx', x(i)); d.setAttribute('cy', y(v)); d.setAttribute('opacity', 1);
        rows += `<div class="r"><i style="background:${col(sr.color)}"></i><span>${esc(sr.name || '')}</span><b>${esc((c.tipFmt || c.yFmt || fmtN)(v, sr))}</b></div>`;
      });
      showTip(`<div class="h">${esc((c.tipX || c.xFmt || String)(labels[i]))}</div>${rows}`, e, off);
    });
    hit.addEventListener('mouseleave', () => { hideTip(hit); off(); });
    el.replaceChildren(svg);
    if (c.legend && series.length > 1) el.append(legend(series));
  };

  function legend(items) {
    const lg = document.createElement('div'); lg.className = 'obc-legend';
    items.forEach(it => { const sp = document.createElement('span'); sp.innerHTML = `<i style="background:${col(it.color)}"></i>${esc(it.name || it.label)}`; lg.append(sp); });
    return lg;
  }

  /* ---------- vertical bars ---------- */
  DRAW.bars = function (el, c) {
    const W = Math.max(innerW(el), 120), H = c.height || 220, rt = c.right;
    const padL = 46, padR = rt ? 56 : 8, padT = rt ? 26 : 12, padB = 26;
    const labels = c.labels || [], n = labels.length;
    const series = (c.series || []).map((sr, i) => Object.assign({ color: PALETTE[i % PALETTE.length] }, sr));
    let hi = 0, lo = 0;
    for (let i = 0; i < n; i++) {
      if (c.stacked) { let p = 0, m = 0; series.forEach(sr => { const v = sr.data[i] || 0; if (v >= 0) p += v; else m += v; }); hi = Math.max(hi, p); lo = Math.min(lo, m); }
      else series.forEach(sr => { const v = sr.data[i] || 0; hi = Math.max(hi, v); lo = Math.min(lo, v); });
    }
    const ticks = niceTicks(lo, hi || 1, 4); lo = ticks[0]; hi = ticks[ticks.length - 1];
    const y = v => padT + (hi - v) / (hi - lo || 1) * (H - padT - padB);
    const bw = (W - padL - padR) / Math.max(1, n);
    const svg = s('svg', { width: W, height: H, class: 'obc' });
    ticks.forEach(t => { svg.append(s('line', { x1: padL, x2: W - padR, y1: y(t), y2: y(t), stroke: css('line'), 'stroke-dasharray': t === 0 ? '' : '2 4' })); const tx = s('text', { x: padL - 8, y: y(t) + 3.5, 'text-anchor': 'end', class: 'obc-ax' }); tx.textContent = (c.yFmt || fmtN)(t); svg.append(tx); });
    // right axis: each left gridline read in the second unit, so a bar's top meets its value on both sides
    let rcol = '';
    if (rt) {
      rcol = col(rt.color || (series[0] && series[0].color));
      ticks.forEach(t => { if (t < 0) return; const tx = s('text', { x: W - padR + 8, y: y(t) + 3.5, 'text-anchor': 'start', class: 'obc-ax' }); tx.textContent = (rt.fmt || fmtN)(t * (rt.per || 0)); svg.append(tx); });
      const axT = (x, a, name, color) => { const tx = s('text', { x, y: 11, 'text-anchor': a, class: 'obc-ax', style: `fill:${color};font-weight:650` }); tx.textContent = name; svg.append(tx); };
      axT(padL - 8, 'end', series.length === 1 ? series[0].name || '' : '', col(series[0] && series[0].color));
      axT(W - padR + 8, 'start', rt.name || '', rcol);
    }
    const every = Math.max(1, Math.ceil(n / Math.max(2, Math.floor((W - padL) / (c.xGap || 56)))));
    const inner = Math.max(2, Math.min(bw * (c.stacked || series.length === 1 ? .62 : .8), c.maxBar || 34));
    for (let i = 0; i < n; i++) {
      const cx = padL + bw * i + bw / 2;
      if (i % every === 0) { const tx = s('text', { x: cx, y: H - 7, 'text-anchor': 'middle', class: 'obc-ax' }); tx.textContent = (c.xFmt || String)(labels[i]); svg.append(tx); }
      const g = s('g', { class: 'obc-bar' });
      let pos = 0, neg = 0;
      series.forEach((sr, si) => {
        const v = sr.data[i] || 0; if (!v && !c.showZero) return;
        let x0, w, y0, y1;
        if (c.stacked) { w = inner; x0 = cx - w / 2; if (v >= 0) { y0 = y(pos + v); y1 = y(pos); pos += v; } else { y0 = y(neg); y1 = y(neg + v); neg += v; } }
        else { w = inner / series.length; x0 = cx - inner / 2 + si * w; y0 = y(Math.max(v, 0)); y1 = y(Math.min(v, 0)); }
        const color = typeof sr.colorFn === 'function' ? col(sr.colorFn(v, i)) : col(sr.color);
        g.append(s('rect', { x: x0 + (series.length > 1 && !c.stacked ? 1 : 0), y: y0, width: Math.max(1, w - (series.length > 1 && !c.stacked ? 2 : 0)), height: Math.max(v ? 1.5 : 0, y1 - y0), rx: Math.min(4, w / 3), fill: color }));
      });
      const hit = s('rect', { x: padL + bw * i, y: padT, width: bw, height: H - padT - padB, fill: 'transparent' });
      const off = () => { g.style.opacity = 1; };
      hit.addEventListener('mousemove', e => {
        g.style.opacity = .82;
        let rows = series.map(sr => `<div class="r"><i style="background:${col(typeof sr.colorFn === 'function' ? sr.colorFn(sr.data[i] || 0, i) : sr.color)}"></i><span>${esc(sr.name || '')}</span><b>${esc((c.tipFmt || c.yFmt || fmtN)(sr.data[i] || 0))}</b></div>`).join('');
        if (rt) { const v = rt.data ? rt.data[i] || 0 : series.reduce((a, sr) => a + (sr.data[i] || 0), 0) * (rt.per || 0); rows += `<div class="r"><i style="background:${rcol}"></i><span>${esc(rt.name || '')}</span><b>${esc((rt.tipFmt || rt.fmt || fmtN)(v))}</b></div>`; }
        showTip(`<div class="h">${esc((c.tipX || c.xFmt || String)(labels[i]))}</div>${rows}`, e, off);
      });
      hit.addEventListener('mouseleave', () => { hideTip(hit); off(); });
      svg.append(g, hit);
    }
    el.replaceChildren(svg);
    if (c.legend && series.length > 1) el.append(legend(series));
  };

  /* ---------- horizontal ranked bars (HTML) ---------- */
  DRAW.hbars = function (el, c) {
    const items = c.items || [], max = c.max || Math.max(1, ...items.map(i => Math.abs(i.value)));
    const wrap = document.createElement('div'); wrap.className = 'obc-hbars';
    items.forEach((it, i) => {
      const row = document.createElement('div'); row.className = 'obc-hrow';
      const pctW = Math.max(1.5, Math.abs(it.value) / max * 100);
      row.innerHTML = `<div class="l"><span class="truncate">${esc(it.label)}</span>${it.sub ? `<em>${esc(it.sub)}</em>` : ''}</div>
        <div class="t"><i style="width:${pctW}%;background:${col(it.color || PALETTE[c.cycle ? i % PALETTE.length : 0])}"></i></div>
        <b class="num">${esc((c.fmt || fmtN)(it.value, it))}</b>`;
      wrap.append(row);
    });
    if (!items.length) wrap.innerHTML = `<div class="obc-empty">${esc(c.empty || 'No data yet')}</div>`;
    el.replaceChildren(wrap);
  };

  /* ---------- donut ---------- */
  DRAW.donut = function (el, c) {
    const size = c.size || 168, r = size / 2, th = c.thickness || 18;
    const items = (c.items || []).filter(i => i.value > 0).map((it, i) => Object.assign({ color: PALETTE[i % PALETTE.length] }, it));
    const total = items.reduce((a, b) => a + b.value, 0) || 1;
    const wrap = document.createElement('div'); wrap.className = 'obc-donut';
    const svg = s('svg', { width: size, height: size, viewBox: `0 0 ${size} ${size}` });
    svg.append(s('circle', { cx: r, cy: r, r: r - th / 2, fill: 'none', stroke: css('surface-3'), 'stroke-width': th }));
    let a0 = -Math.PI / 2;
    const gap = items.length > 1 ? 0.03 : 0;
    items.forEach(it => {
      const a1 = a0 + it.value / total * Math.PI * 2;
      const rr = r - th / 2, s0 = a0 + gap / 2, s1 = Math.max(s0 + 0.001, a1 - gap / 2);
      const large = s1 - s0 > Math.PI ? 1 : 0;
      const p = s('path', { d: `M${r + rr * Math.cos(s0)},${r + rr * Math.sin(s0)}A${rr},${rr} 0 ${large} 1 ${r + rr * Math.cos(s1)},${r + rr * Math.sin(s1)}`, fill: 'none', stroke: col(it.color), 'stroke-width': th, 'stroke-linecap': items.length > 1 ? 'butt' : 'round' });
      p.addEventListener('mousemove', e => showTip(`<div class="h">${esc(it.label)}</div><div class="r"><i style="background:${col(it.color)}"></i><span>${(it.value / total * 100).toFixed(1)}%</span><b>${esc((c.fmt || fmtN)(it.value))}</b></div>`, e));
      p.addEventListener('mouseleave', () => hideTip(p));
      svg.append(p); a0 = a1;
    });
    const center = document.createElement('div'); center.className = 'obc-donut-c';
    if (c.center) center.innerHTML = `<b class="num">${esc(c.center.value)}</b><span>${esc(c.center.label || '')}</span>`;
    const holder = document.createElement('div'); holder.className = 'obc-donut-svg'; holder.style.width = holder.style.height = size + 'px'; holder.append(svg, center);
    wrap.append(holder);
    if (c.legend !== false) {
      const lg = document.createElement('div'); lg.className = 'obc-dlegend';
      items.forEach(it => { const d = document.createElement('div'); d.innerHTML = `<i style="background:${col(it.color)}"></i><span>${esc(it.label)}</span><b class="num">${esc((c.fmt || fmtN)(it.value))}</b><em>${(it.value / total * 100).toFixed(0)}%</em>`; lg.append(d); });
      wrap.append(lg);
    }
    el.replaceChildren(wrap);
  };

  /* ---------- sparkline ---------- */
  DRAW.spark = function (el, c) {
    const data = (c.data || []).filter(v => v != null);
    const W = Math.max(innerW(el), 40), H = c.height || 36;
    const svg = s('svg', { width: W, height: H, class: 'obc' });
    if (data.length > 1) {
      const lo = Math.min(...data), hi = Math.max(...data);
      const x = i => i * (W - 2) / (data.length - 1) + 1, y = v => 2 + (hi - v) / (hi - lo || 1) * (H - 4);
      const color = col(c.color || (data[data.length - 1] >= data[0] ? 'bull' : 'bear'));
      const d = data.map((v, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ',' + y(v).toFixed(1)).join('');
      if (c.area !== false) {
        const gid = 'sg' + Math.random().toString(36).slice(2, 8); const defs = s('defs'); const g = s('linearGradient', { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 });
        g.append(s('stop', { offset: '0%', 'stop-color': color, 'stop-opacity': .25 }), s('stop', { offset: '100%', 'stop-color': color, 'stop-opacity': 0 })); defs.append(g); svg.append(defs);
        svg.append(s('path', { d: d + `L${x(data.length - 1)},${H}L${x(0)},${H}Z`, fill: `url(#${gid})` }));
      }
      svg.append(s('path', { d, fill: 'none', stroke: color, 'stroke-width': 1.6, 'stroke-linejoin': 'round' }));
    }
    el.replaceChildren(svg);
  };

  /* ---------- calendar heatmap (diverging: profit green / loss red) ---------- */
  DRAW.calendar = function (el, c) {
    const vals = c.values || {};
    const to = c.to ? new Date(c.to + 'T00:00:00') : new Date();
    const from = c.from ? new Date(c.from + 'T00:00:00') : new Date(to.getTime() - 364 * 864e5);
    const start = new Date(from); start.setDate(start.getDate() - ((start.getDay() + 6) % 7)); // Monday
    const weeks = Math.ceil(((to - start) / 864e5 + 1) / 7);
    // cells fit the width; on a phone they stay big enough to read and tap, and a long range
    // scrolls sideways (opening on the newest weeks) with the day names pinned on the left
    const gap = 3, L = 28, W = Math.max(innerW(el), 200);
    const cell = Math.max(W < 520 ? 12 : 8, Math.min(c.maxCell || 16, Math.floor((W - L - gap * weeks) / weeks)));
    const H = 18 + 7 * (cell + gap), GW = weeks * (cell + gap);
    const days = s('svg', { width: L, height: H, class: 'obc' });
    const svg = s('svg', { width: GW, height: H, class: 'obc' });
    const nums = Object.values(vals).filter(v => typeof v === 'number');
    const maxAbs = Math.max(1, ...nums.map(Math.abs));
    const pos = col(c.pos || 'bull'), neg = col(c.neg || 'bear'), empty = css('surface-3');
    ['Mon', '', 'Wed', '', 'Fri', '', ''].forEach((d, i) => { if (!d) return; const t = s('text', { x: 0, y: 18 + i * (cell + gap) + cell - 2, class: 'obc-ax' }); t.textContent = d; days.append(t); });
    let lastMonth = -1;
    const ymd = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    for (let w = 0; w < weeks; w++) for (let d = 0; d < 7; d++) {
      const day = new Date(start.getTime() + (w * 7 + d) * 864e5);
      if (day < from || day > to) continue;
      const key = ymd(day), v = vals[key];
      const x = w * (cell + gap), y = 18 + d * (cell + gap);
      if (day.getDate() <= 7 && day.getMonth() !== lastMonth && d === 0) { lastMonth = day.getMonth(); const end = x + 24 > GW; const t = s('text', { x: end ? GW : x, y: 10, 'text-anchor': end ? 'end' : 'start', class: 'obc-ax' }); t.textContent = day.toLocaleString('en-IN', { month: 'short' }); svg.append(t); }
      let fill = empty, op = 1;
      if (typeof v === 'number' && v !== 0) { fill = v > 0 ? pos : neg; op = .25 + .75 * Math.min(1, Math.sqrt(Math.abs(v) / maxAbs)); }
      else if (v === 0) { fill = css('ink-4'); op = .35; }
      const r = s('rect', { x, y, width: cell, height: cell, rx: Math.min(4, cell / 3.5), fill, 'fill-opacity': op });
      r.addEventListener('mousemove', e => showTip(c.tip ? c.tip(key, v) : `<div class="h">${esc(day.toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' }))}</div><div class="r"><b>${v == null ? 'No trades' : esc((c.fmt || fmtN)(v))}</b></div>`, e));
      r.addEventListener('mouseleave', () => hideTip(r));
      if (c.onClick) { r.style.cursor = 'pointer'; r.addEventListener('click', () => c.onClick(key, v)); }
      svg.append(r);
    }
    const wrap = document.createElement('div'); wrap.className = 'obc-calw';
    const box = document.createElement('div'); box.className = 'obc-cal'; box.append(svg);
    wrap.append(days, box);
    el.replaceChildren(wrap);
    const edges = () => { const m = box.scrollWidth - box.clientWidth; box.classList.toggle('fl', box.scrollLeft > 2); box.classList.toggle('fr', box.scrollLeft < m - 2); };
    if (box.scrollWidth > box.clientWidth) box.scrollLeft = box.scrollWidth;
    box.addEventListener('scroll', edges, { passive: true });
    edges();
  };

  /* ---------- matrix heatmap (e.g. weekday x hour) ---------- */
  DRAW.matrix = function (el, c) {
    const rows = c.rows || [], cols = c.cols || [], vals = c.values || [];
    const W = Math.max(innerW(el), 200), padL = 40, gap = 3;
    const cw = Math.max(6, (W - padL - gap * cols.length) / cols.length), ch = c.cellH || Math.min(26, Math.max(14, cw * .8));
    const H = 18 + rows.length * (ch + gap);
    const svg = s('svg', { width: W, height: H, class: 'obc' });
    let mx = 0; vals.forEach(r => (r || []).forEach(v => mx = Math.max(mx, v || 0)));
    const color = col(c.color || 'brand'), empty = css('surface-3');
    cols.forEach((cl, j) => { if (c.colEvery && j % c.colEvery) return; const t = s('text', { x: padL + j * (cw + gap) + cw / 2, y: 10, 'text-anchor': 'middle', class: 'obc-ax' }); t.textContent = cl; svg.append(t); });
    rows.forEach((rw, i) => {
      const t = s('text', { x: 0, y: 18 + i * (ch + gap) + ch / 2 + 3.5, class: 'obc-ax' }); t.textContent = rw; svg.append(t);
      cols.forEach((cl, j) => {
        const v = (vals[i] || [])[j] || 0;
        const r = s('rect', { x: padL + j * (cw + gap), y: 18 + i * (ch + gap), width: cw, height: ch, rx: 3.5, fill: v ? color : empty, 'fill-opacity': v ? .12 + .88 * Math.pow(v / (mx || 1), .7) : 1 });
        r.addEventListener('mousemove', e => showTip(`<div class="h">${esc(rw)} · ${esc(c.colLabel ? c.colLabel(cl) : cl)}</div><div class="r"><b>${esc((c.fmt || fmtN)(v))}</b></div>`, e));
        r.addEventListener('mouseleave', () => hideTip(r));
        svg.append(r);
      });
    });
    el.replaceChildren(svg);
  };

  /* ---------- funnel ---------- */
  DRAW.funnel = function (el, c) {
    const steps = c.steps || [], top = steps[0] ? steps[0].value || 1 : 1;
    const wrap = document.createElement('div'); wrap.className = 'obc-funnel';
    steps.forEach((st, i) => {
      const p = st.value / top * 100, prev = i ? steps[i - 1].value || 1 : null;
      const conv = prev != null ? (st.value / prev * 100) : null;
      const row = document.createElement('div'); row.className = 'obc-fstep';
      row.innerHTML = `<div class="fl"><b>${esc(st.label)}</b>${conv != null ? `<em class="${conv >= 50 ? 'up' : conv >= 20 ? '' : 'down'}">${conv.toFixed(1)}% from previous</em>` : '<em>Start</em>'}</div>
        <div class="fb"><i style="width:${Math.max(1, p)}%;background:${col(st.color || PALETTE[i % PALETTE.length])}"></i></div>
        <div class="fv"><b class="num">${esc((c.fmt || fmtN)(st.value))}</b><em>${p.toFixed(p < 10 ? 1 : 0)}%</em></div>`;
      wrap.append(row);
    });
    el.replaceChildren(wrap);
  };

  window.OBC = {
    line: (el, c) => mount(el, 'line', c),
    bars: (el, c) => mount(el, 'bars', c),
    hbars: (el, c) => mount(el, 'hbars', c),
    donut: (el, c) => mount(el, 'donut', c),
    spark: (el, data, c = {}) => mount(el, 'spark', Object.assign({ data }, c)),
    calendar: (el, c) => mount(el, 'calendar', c),
    matrix: (el, c) => mount(el, 'matrix', c),
    funnel: (el, c) => mount(el, 'funnel', c),
    color: col,
    tip: showTip,
    hideTip,
  };
})();
