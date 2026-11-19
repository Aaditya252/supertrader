"use client";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
type Candidate = {
  id: string; source: string; title: string; url: string; summary: string; license: string;
  observed_at: string; families: string[]; review_gaps: string[]; content_hash: string;
  content_basis: string; assessment: string;
};
type Comparison = { status: string; version?: string; approved?: boolean; promotion_failures?: string[];
  rows: { name: string; metrics: Record<string, number | null> }[] };
const key = "supertrader-strategy-watchlist-v1";
const percent = (value: number | null | undefined) => value == null ? "Unavailable" : `${(value * 100).toFixed(2)}%`;

export default function StrategyFinder() {
  const [query, setQuery] = useState("SOL momentum");
  const [source, setSource] = useState("github");
  const [items, setItems] = useState<Candidate[]>([]);
  const [saved, setSaved] = useState<Candidate[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [comparisonError, setComparisonError] = useState("");
  useEffect(() => {
    try { const value = JSON.parse(localStorage.getItem(key) ?? "[]");
      if (Array.isArray(value)) setSaved(value.filter((v) => v && typeof v.id === "string" && typeof v.url === "string"
        && typeof v.title === "string" && typeof v.summary === "string" && Array.isArray(v.families)
        && v.families.every((f: unknown) => typeof f === "string") && Array.isArray(v.review_gaps)
        && v.review_gaps.every((f: unknown) => typeof f === "string") && typeof v.observed_at === "string"
        && typeof v.license === "string" && typeof v.content_basis === "string" && typeof v.assessment === "string").slice(0, 50));
    } catch { setError("Saved watchlist could not be loaded."); }
    fetch(API + "/strategies/comparison", { signal: AbortSignal.timeout(10000) })
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); }).then(setComparison)
      .catch(() => setComparisonError("Research comparison unavailable. Check backend connectivity."));
  }, []);
  async function search() {
    setBusy(true); setError(""); setItems([]); setSearched(true);
    try {
      const response = await fetch(API + "/strategies/discover?" + new URLSearchParams({ q: query, source }), { signal: AbortSignal.timeout(80000) });
      if (!response.ok) throw new Error(response.status === 422 ? "Use 2–100 characters: letters, numbers, spaces, dots, hyphens or underscores." : "Search failed. Check backend connectivity.");
      const data = await response.json();
      if (data.status !== "ready") throw new Error(data.error + (data.retry_after_seconds ? ` Retry in ${data.retry_after_seconds}s.` : ""));
      setItems(data.items);
    } catch (e) { setError(e instanceof Error ? e.message : "Provider unavailable."); }
    finally { setBusy(false); }
  }
  function toggle(item: Candidate) {
    const next = saved.some((s) => s.id === item.id) ? saved.filter((s) => s.id !== item.id) : [...saved, item].slice(-50);
    try { localStorage.setItem(key, JSON.stringify(next)); setSaved(next); }
    catch { setError("Browser storage unavailable; watchlist was not saved."); }
  }
  function card(item: Candidate) {
    const safe = /^https:\/\/(github\.com|arxiv\.org)\//.test(item.url);
    return <article className="strategy-card" key={item.id}>
      <div className="panel-heading"><h3>{item.title}</h3><span className="badge amber">UNVALIDATED</span></div>
      <p>{item.summary || "No description available."}</p>
      <p className="muted">{item.families.join(" · ") || "Unclassified"} · {item.content_basis}</p>
      <p className="footnote">License: {item.license}. Collected: {new Date(item.observed_at).toLocaleString()}.</p>
      <p className="footnote">Review gaps: {item.review_gaps.join(", ") || "Exact rules still need review"}. {item.assessment}</p>
      <div className="strategy-actions">{safe && <a href={item.url} target="_blank" rel="noopener noreferrer">Read original source ↗</a>}
        <button onClick={() => toggle(item)}>{saved.some((s) => s.id === item.id) ? "Remove from watchlist" : "Save for review"}</button></div>
    </article>;
  }
  return <div className="strategy-finder">
    <section className="panel"><div className="panel-heading"><h2>Live strategy finder</h2><span className="badge neutral">RESEARCH</span></div>
      <div className="panel-body"><p>Discover public repositories and research abstracts. Results include source provenance, license information and rule-review gaps. Every candidate needs independent testing.</p>
        <form className="strategy-search" onSubmit={(e) => { e.preventDefault(); void search(); }}>
          <label>Strategy topic<input value={query} onChange={(e) => setQuery(e.target.value)} minLength={2} maxLength={100} required placeholder="SOL momentum" /></label>
          <label>Source<select value={source} onChange={(e) => setSource(e.target.value)}><option value="github">GitHub repositories + README</option><option value="arxiv">arXiv research abstracts</option></select></label>
          <button disabled={busy} type="submit">{busy ? "Searching sources…" : "Find strategies"}</button>
        </form><p className="footnote">On-demand searches use a 10-minute cache. Source content is never executed. Keyword coverage is a review aid, not a quality score.</p>
        {error && <p role="alert" className="red-text">{error}</p>}
        {!busy && searched && !error && !items.length && <p>No matching candidates. Try a broader topic.</p>}
        <div aria-live="polite">{items.map(card)}</div>
      </div></section>
    <section className="panel spaced"><div className="panel-heading"><h2>Research watchlist</h2><span>{saved.length} / 50</span></div>
      <div className="panel-body"><p className="muted">Saved in this browser. Discovery date does not establish historical availability for a backtest.</p>
        {saved.length ? saved.map(card) : <p>Save a candidate to build your review queue.</p>}</div></section>
    <section className="panel spaced"><div className="panel-heading"><h2>Local benchmark comparison</h2><span className="badge neutral">{comparison?.approved ? "HISTORICAL GATES PASSED" : "RESEARCH ONLY"}</span></div>
      <div className="panel-body"><p>Existing local experiment: {comparison?.version ?? "No report"}. Web discoveries have no backtest results yet.</p>
        {comparisonError && <p role="alert" className="red-text">{comparisonError}</p>}
        {comparison?.promotion_failures?.map((reason) => <p className="amber-text" key={reason}>{reason}</p>)}</div>
      <div className="table-wrap"><table><thead><tr><th>Strategy</th><th>Net return</th><th>Sharpe</th><th>Max drawdown</th><th>Trades</th></tr></thead>
        <tbody>{comparison?.rows.map((row) => <tr key={row.name}><td>{row.name.replaceAll("_", " ")}</td><td>{percent(row.metrics.total_return)}</td><td>{row.metrics.sharpe?.toFixed(2) ?? "Unavailable"}</td><td>{percent(row.metrics.max_drawdown)}</td><td>{row.metrics.trade_count ?? "Unavailable"}</td></tr>)}</tbody></table>
        {comparison?.status === "not_run" && <p className="panel-body">Train a local experiment to populate this comparison.</p>}</div>
    </section>
  </div>;
}
