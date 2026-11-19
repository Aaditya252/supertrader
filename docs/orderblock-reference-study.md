# OrderBlock reference study and Supertrader implementation

Inspected 27 October 2026. Target: https://order-block.com/app/orderblock. Browser: the user's existing Brave guest session. This study concerns the terminal and its stations, not a completed clone of the journal, learning, account, community or marketing pages.

## Current delivery

`frontend/app/market-flow/page.tsx` adds a functional, independently authored design draft to Supertrader. It is accessible through the existing terminal's Market Flow navigation link. It uses a white canvas, green/red paths, sector breadth, ranked movers, a search dialog, stock details, browser-local watchlists, CSV export and a coherent replay playhead.

The subsequent pasted brief requires a literal 1:1 frontend reproduction. The existing draft does **not** meet that requirement. It has been preserved as a reviewable draft rather than represented as the final clone. No backend feed, exchange data licence, paid access or institutional-signal calculation has been added.

## Observed reference structure

The actual terminal is embedded in a same-origin iframe. Desktop uses an absolutely positioned 1920-by-1200 scene scaled into the available viewport. Its layers include multiple canvases for axes, trajectories, walls, ladder, connecting paths and effects; positioned sector chips; ranked tiles; six larger setup tickets; search; and an inspector. The phone/tablet version replaces that scene with stacked sections and a sticky station navigation bar.

The accessible account is a guest sample. The shell displays an access lock, and the sample intentionally blurs portions of the content. Those restrictions were not removed. Stock search exposes 210 illustrative entries, saved separately from the authored draft's synthetic 69-stock universe.

The chart represents session-relative price trajectories. It is not evidence that identified cash moved directly from one security to another. Sector participation and ranked directional moves provide context.

## Disclosed calculations

The [disclosures](https://order-block.com/disclosures) describe a strength measure based on a stock's directional move from the session open relative to its typical daily movement. Their explanation uses roughly 50 calendar days of completed history, applies a one-sided-move eligibility rule, and refreshes rankings each minute. These statements describe the reference's disclosure, not verified source code or a calculation implemented in the current draft. The exact liquidity and order-block engine remains unverified.

The [updates page](https://order-block.com/updates) documents minute refreshes, mobile stock sheets, exchange-calendar handling and historical-price adjustments. These are useful production requirements; they do not establish the private server's framework, storage or operational reliability.

## Measured design evidence

See `orderblock-capture/design-tokens.json` for actual CSSOM rules rather than rounded design estimates, and `orderblock-capture/frame-styles.css` for extracted frame styles. The observed font families include Inter Tight, JetBrains Mono and Barlow Condensed. These fonts differ from the draft's Arial-based styling.

Reference light/dark screenshots exist at 320, 375, 414, 768, 1024, 1280, 1440 and 1920 pixels. Capture records contain viewport geometry, rendered markup, computed properties and accessible stylesheets. Screenshot dimensions and interaction states must be checked before accepting a pair as a valid visual-comparison baseline; some later experimental screenshots were inconsistent and are not accepted as fidelity proof.

## Clean data boundary

The authored draft's content is in `frontend/data/market-flow-preview.ts`. `frontend/lib/api/market-flow.ts` is the single snapshot adapter. Its returned object explicitly identifies synthetic preview mode, local fixture provenance, Asia/Kolkata timezone and five-minute sample intervals. `frontend/data/orderblock-reference.json` stores the separate guest-search observations; it is not a live market-data response.

To connect a real backend, replace the adapter and evolve its typed response. Do not silently relabel the fixture as live data. A production response needs timestamped bars, real session identifiers, quality/freshness information and explicit missing-value handling; the current compact fixture is insufficient as a production transport contract.

## Proposed production system — architecture, not an observed backend

1. A market-data adapter receives a licensed Indian-equity feed, or an explicitly chosen crypto feed for Supertrader's existing crypto scope. Provider credentials stay server-side.
2. Instrument and sector mappings use stable provider identifiers, effective dates and corporate-action adjustments. Define the real eligible universe rather than assuming the sample is the exchange's current F&O list.
3. An exchange calendar determines the actual session, holidays and special sessions. Store UTC timestamps; display the appropriate exchange timezone.
4. Ingestion validates sequences, timestamps, prices and volumes, deduplicates updates, detects gaps and backfills missing history. Keep raw observations and completed-bar status.
5. An aggregation service calculates explicitly defined returns, breadth, sector measures and any separately specified strength model using only data available at the corresponding time. Do not present inferred institutional intent as an observed fact.
6. Persist session history and computed snapshots. A reconnect obtains a canonical snapshot before consuming incremental updates. Never display stale prices as live.
7. A REST snapshot plus SSE/WebSocket updates supplies the frontend. Include event sequence, server time, observation time, stale/partial status and a revision identifier. Refresh without resetting the user's selected stock, filters or scroll.
8. A separately specified, tested zone engine can calculate order-block candidates from completed candles. Track creation, retest, invalidation and the information available at formation. The reference's proprietary rules have not been recovered.

## Verification performed

- Initial `npm.cmd run typecheck`: passed.
- `npm.cmd run build -- --webpack`: passed, including TypeScript and static generation for `/`, `/_not-found` and `/market-flow`.
- Default Turbopack production build: failed while starting a CSS worker, including on an existing stylesheet. The supported webpack fallback succeeded.
- Browser: search for SBIN, detail values, watchlist action, Banking sector drill-down, replay start returning zero moves, reset to close, and 20-entry bullish board verified.
- Browser CSV export was visible as a completed 4.3 KB download. The download-event wait timed out; file contents were not independently parsed.
- Brave logged extension-injected wallet errors and hydration differences involving extension-added `bis_*` attributes. No source change was made to conceal those extension modifications.
- Next.js emitted a security upgrade advisory for the installed version. No dependency upgrade was performed as part of this visual task.

## Fidelity report

**Status: FAIL / incomplete against the 1:1 brief.** There is no valid near-zero pixel-diff result, no completed element-for-element style equivalence, no frame-by-frame animation verification, and no Lighthouse result. No percentages or scores are invented.

Known differences: shell and navigation layout; page heading; font family; spacing; 69 synthetic stocks versus 210 sample entries; 12 versus 21 sectors; SVG paths versus the reference's canvas layers; different chart histories and ranking semantics; absent exact desktop connectors and setup-ticket layouts; no reference dark theme; and no cloned journal/account/community pages.

Required to finish the exact canvas reproduction without inventing values: an authorized frontend source/export containing the renderer and its sample drawing data, or a permitted capture mechanism exposing those exact drawing inputs and animation timings. To match an unblurred paid screen, access supplied by the user is also required. The guest sample cannot establish that screen's appearance.

The browser rejected page-source protocol access. No alternate protocol, session-cookie export or hidden-state workaround was used. Public unauthenticated requests for the terminal sample returned login HTML and were not treated as a successful source capture.

## Run the current draft

From `D:\supertrader\frontend`:

```powershell
npm.cmd ci
npm.cmd run dev -- --port 3010
```

Open `http://127.0.0.1:3010/market-flow`. For a verified production build, use `npm.cmd run build -- --webpack`, followed by `npm.cmd run start -- --port 3010`.

Complete authored files are provided in `output/market-flow-complete-source.md` and `output/supertrader-market-flow-source.zip`. Existing Supertrader source remains in the workspace. The draft is local-only. Reference assets and branding are study evidence and have not been deployed.
