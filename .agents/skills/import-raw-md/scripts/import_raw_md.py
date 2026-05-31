"""Import raw Markdown into the project's organized docs folders."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DESTINATIONS = {
    "wiki": Path("docs/wiki"),
    "safe": Path("docs/SAFe"),
}


def repo_root() -> Path:
    return Path.cwd().resolve()


def resolve_under_root(path: Path) -> Path:
    root = repo_root()
    resolved = (root / path).resolve()
    resolved.relative_to(root)
    return resolved


def infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return fallback


def pascal_slug(title: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", title)
    if not words:
        raise ValueError("could not derive a destination file name")
    return "".join(word[:1].upper() + word[1:] for word in words) + ".md"


def ensure_h1(text: str, title: str) -> str:
    if re.search(r"^#\s+", text, flags=re.MULTILINE):
        return text.rstrip() + "\n"
    return f"# {title}\n\n{text.strip()}\n"


def update_wiki_index(dest_rel: Path, title: str, dry_run: bool) -> None:
    if dest_rel.parent.as_posix() != "docs/wiki":
        return

    index = resolve_under_root(Path("docs/wiki/README.md"))
    link = f"- [{title}]({dest_rel.name})"
    existing = index.read_text(encoding="utf-8") if index.exists() else "# Wuzhiqi AlphaZero Wiki\n"
    if link in existing:
        return

    next_text = existing.rstrip() + "\n" + link + "\n"
    if dry_run:
        print(f"would update {index.relative_to(repo_root())} with {link}")
        return
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(next_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Markdown file under docs/raw")
    parser.add_argument("--dest", choices=DESTINATIONS, default="wiki")
    parser.add_argument("--title", help="Override page title")
    parser.add_argument("--name", help="Override destination file name")
    parser.add_argument("--move", action="store_true", help="Remove the raw source after import")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_under_root(args.source)
    raw_root = resolve_under_root(Path("docs/raw"))
    source.relative_to(raw_root)

    if source.suffix.lower() != ".md":
        raise ValueError("source must be a Markdown file")

    text = source.read_text(encoding="utf-8")
    title = args.title or infer_title(text, source.stem)
    filename = args.name or pascal_slug(title)
    if not filename.endswith(".md"):
        filename += ".md"

    dest_dir = resolve_under_root(DESTINATIONS[args.dest])
    destination = dest_dir / filename
    dest_rel = destination.relative_to(repo_root())
    output = ensure_h1(text, title)

    if destination.exists():
        raise FileExistsError(f"destination already exists: {dest_rel}")

    if args.dry_run:
        print(f"would import {source.relative_to(repo_root())} -> {dest_rel}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")
        print(f"imported {source.relative_to(repo_root())} -> {dest_rel}")

    update_wiki_index(dest_rel, title, args.dry_run)

    if args.move:
        if args.dry_run:
            print(f"would remove {source.relative_to(repo_root())}")
        else:
            source.unlink()


if __name__ == "__main__":
    main()
