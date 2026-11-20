# OrderBlock capture inventory

Target: https://order-block.com/app/orderblock. Captured from the user's accessible Brave guest sample on 27 October 2026. This folder is evidence for a local frontend study; it is not an unblurred paid-terminal export or a completed clone.

## Files

- `reference-light-WIDTH.jpg` and `reference-dark-WIDTH.jpg`: screenshots at the eight requested widths, 900-pixel viewport height. The embedded terminal scroll position can differ between captures. They are not approved same-state pixel-diff pairs.
- `light-WIDTH.json` and `dark-WIDTH.json`: rendered outer/frame markup in chunks, element measurements, computed styles and available CSSOM rules. Script bodies are excluded from DOM captures. Attribute values can reflect extensions installed in the browser.
- `frame-styles.css`: frame styles extracted from the accessible CSSOM.
- `design-tokens.json`: exact observed root/theme, responsive and animation rules, plus font families. It is source evidence, not an approximation to Tailwind values.
- `frame-rendered.html`, `outer-rendered.html`: reconstructed captured markup for inspection. Canvas drawing contents are not serialized by HTML.
- `stock-search-data.json`: 210 visible search-option strings from the guest sample. These are illustrative sample identities and values.
- `assets-manifest.json`: observed stylesheet/image asset export results and paths. Exported assets are retained locally under `assets/` where available. One SVG sprite export failed content-type validation.
- `public-source/`: public frontend assets obtained without credentials. `sample.html` is actually a sign-in redirect response, not the terminal source. It must not be treated as successful terminal acquisition.
- `draft-light-WIDTH.jpg`: exploratory draft screenshots. Several bitmap sizes differ from the requested viewport; these are unsuitable for a valid paired fidelity percentage.
- `flow-reference-light-WIDTH.jpg`: experimental captures whose viewport settings applied to a different selected tab. These are not valid per-width baselines.

## Coverage and limits

The browser returned at most 2000 element measurements per document. Very large markup strings were subsequently captured in chunks, but earlier `frame-light.json` is truncated. None of this is presented as a complete measurement of every node and pseudo-element.

Canvas extraction and the private API's response payloads were unavailable through the supported browser tools. No authentication cookies were exported, no paid restrictions removed and no private server implementation established. Page-source protocol navigation was rejected by browser policy; no workaround was attempted.

The site visibly uses CSS animations and canvas rendering. No specific animation library is claimed without evidence. Full hover/focus/disabled/error/empty-state matrices, animation timings for canvas draws, authenticated non-terminal pages, pixelmatch, style equivalence and Lighthouse checks remain incomplete.

## Fidelity status

| Width | Light reference | Dark reference | Accepted same-state pixel diff |
|---:|---|---|---|
| 320 | Captured | Captured | Not measured |
| 375 | Captured | Captured | Not measured |
| 414 | Captured | Captured | Not measured |
| 768 | Captured | Captured | Not measured |
| 1024 | Captured | Captured | Not measured |
| 1280 | Captured | Captured | Not measured |
| 1440 | Captured | Captured | Not measured |
| 1920 | Captured | Captured | Not measured |

The authored draft fails the literal 1:1 requirement. See `../orderblock-reference-study.md` for specific differences, verified application behavior, setup instructions and the exact missing evidence.
