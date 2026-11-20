"""Inspect saved, rendered UI evidence. Never loads credentials or remote state."""
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "docs" / "orderblock-capture"


class Element:
    def __init__(self, tag, attrs, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children = []


class Tree(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, html):
        super().__init__(convert_charrefs=False)
        self.root = Element("root", [])
        self.current = self.root
        self.ids = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = Element(tag, attrs, self.current)
        self.current.children.append(node)
        if node.attrs.get("id"):
            self.ids[node.attrs["id"]] = node
        if tag not in self.VOID:
            self.current = node

    def handle_endtag(self, tag):
        node = self.current
        while node.parent and node.tag != tag:
            node = node.parent
        if node.parent:
            self.current = node.parent

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)


if __name__ == "__main__":
    for width in [375, 1440]:
        record = json.loads((CAPTURE / f"light-{width}.json").read_text(encoding="utf-8"))
        html = "".join(record["frame"]["domChunks"])
        tree = Tree(html)
        print("\nWIDTH", width, "LENGTH", len(html))
        for id_ in ["stage", "plumeWrap", "cvPlume", "cvAxis", "cvLadder", "stack", "phNav", "obFind"]:
            node = tree.ids.get(id_)
            if node:
                print(id_, node.attrs, "children", [(x.tag, x.attrs.get("id"), x.attrs.get("class")) for x in node.children[:14]])
