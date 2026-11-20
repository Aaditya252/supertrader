"""Package local design evidence and the complete authored preview source."""
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "docs" / "orderblock-capture"


def separate_fixture():
    module = ROOT / "frontend/lib/market-flow.ts"
    source = module.read_text(encoding="utf-8")
    if "const universe:" not in source:
        return
    start, end = source.index("const universe:"), source.index("\nexport const moveAt=")
    data = ROOT / "frontend/data"
    data.mkdir(exist_ok=True)
    (data / "market-flow-preview.ts").write_text(
        'import type { FlowAsset } from "@/lib/market-flow";\n'
        'const SESSION_STEPS = 75;\n' + source[start:end] + "\n", encoding="utf-8"
    )
    module.write_text(source[:start] + source[end:], encoding="utf-8")
    component = ROOT / "frontend/components/market-flow/MarketFlow.tsx"
    text = component.read_text(encoding="utf-8").replace("{DEMO_ASSETS,FlowAsset,", "{FlowAsset,")
    text = text.replace('import FlowCanvas from "./FlowCanvas";', 'import FlowCanvas from "./FlowCanvas";\nimport {getMarketFlowSnapshot} from "@/lib/api/market-flow";\nconst DEMO_ASSETS = getMarketFlowSnapshot().assets;')
    component.write_text(text, encoding="utf-8")


def extract_tokens():
    record = json.loads((CAPTURE / "light-1440.json").read_text(encoding="utf-8"))
    rules = [r for sheet in record["frame"]["styles"] for r in sheet.get("rules", [])]
    tokens = {
        "source": "Computed browser CSSOM, accessible guest sample; not inferred values",
        "rootAndThemeRules": [r for r in rules if r.startswith(":root") or r.startswith("html.obl") or r.startswith(".obl")],
        "responsiveRules": [r for r in rules if r.startswith("@media") or r.startswith("@container")],
        "animationRules": [r for r in rules if r.startswith("@keyframes") or r.startswith("@property")],
        "observedFontFamilies": sorted({e["style"]["fontFamily"] for e in record["frame"]["elements"]}),
        "captureLimit": "The browser returned at most 2000 element measurements per document. Canvas pixels and source draw calls were not exposed.",
    }
    (CAPTURE / "design-tokens.json").write_text(json.dumps(tokens, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = json.loads((CAPTURE / "stock-search-data.json").read_text(encoding="utf-8"))
    stocks = []
    for row in rows:
        match = re.match(r"([A-Z]+)([A-Z][a-z].*?) · (.*?)([+−-]\d+\.\d+)%([A-Z ]+?)(?: #(\d+))?$", row)
        if match:
            symbol, name, sector, change, category, rank = match.groups()
            stocks.append(dict(symbol=symbol, name=name, sector=sector, changePercent=float(change.replace("−", "-")), category=category.strip(), rank=int(rank) if rank else None))
        else:
            stocks.append({"visibleText": row})
    data = ROOT / "frontend/data"
    data.mkdir(exist_ok=True)
    (data / "orderblock-reference.json").write_text(json.dumps({"mode": "captured_guest_sample", "source": "Visible stock-search options", "stocks": stocks}, indent=2, ensure_ascii=False), encoding="utf-8")


def package():
    files = [
        "frontend/app/market-flow/page.tsx", "frontend/app/market-flow/market-flow.css",
        "frontend/components/market-flow/MarketFlow.tsx", "frontend/components/market-flow/FlowCanvas.tsx",
        "frontend/data/market-flow-preview.ts", "frontend/data/orderblock-reference.json",
        "frontend/lib/market-flow.ts", "frontend/lib/api/market-flow.ts",
        "frontend/package.json", "frontend/package-lock.json", "frontend/tsconfig.json",
        "docs/orderblock-capture/README.md", "docs/orderblock-reference-study.md",
    ]
    with zipfile.ZipFile(ROOT / "output/supertrader-market-flow-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in files:
            path = ROOT / filename
            if path.is_file():
                archive.write(path, filename)
    contents = ["# Complete authored Market Flow source", "", "This is the functional design draft. It is not a verified 1:1 clone.", "", "```text", *files, "```", ""]
    for name in files[:8]:
        contents.extend([f"## {name}", "", "```" + ("css" if name.endswith(".css") else "json" if name.endswith(".json") else "tsx"), (ROOT / name).read_text(encoding="utf-8"), "```", ""])
    (ROOT / "output/market-flow-complete-source.md").write_text("\n".join(contents), encoding="utf-8")


if __name__ == "__main__":
    separate_fixture()
    extract_tokens()
    package()
    print("Data adapter, measured tokens, captured stock rows and source package saved.")
