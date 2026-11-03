#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCS 公共 ACMI 批量下载/整理工具。

来源（公开目录索引，非官方）：
    http://jedi.gildia.org/acmi/

默认只列出，不下载。真正下载需显式 --download。
仅用于本项目研究/战术动作分析；请遵守站点条款，不要重新分发原始 ACMI。

示例：
    python fetch_dcs_acmi.py list --include "bvr|BFM|F-16|F16|F-18|F18"
    python fetch_dcs_acmi.py list --since 2026-09-01 --until 2026-12-31
    python fetch_dcs_acmi.py download --include "bvr|BFM|F-16|F16|F-18|F18" --max-size-mb 80
    python fetch_dcs_acmi.py import "新增资料"

    # 已有 32 个 ACMI 复制到新目录并生成 manifest（不删除源文件）
    python fetch_dcs_acmi.py import "新增资料" --copy

目录：
    飞行数据集资料库/05_人类战术动作与招式库/03_仿真与ACMI/DCS_BVR/
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT_URL = "http://jedi.gildia.org/acmi/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
DATE_RE = re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})")
HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.I)
ACMI_RE = re.compile(r"\.acmi$", re.I)
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "03_仿真与ACMI" / "DCS_BVR"


@dataclass
class Entry:
    url: str
    name: str
    date: str = ""
    size: int | None = None
    status: str = "listed"


def _request(url: str, *, method: str = "GET", timeout: float = 30,
             headers: dict | None = None):
    hdrs = {"User-Agent": UA, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, method=method, headers=hdrs)
    return urllib.request.urlopen(req, timeout=timeout)


def _fetch_text(url: str, timeout: float) -> str:
    with _request(url, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, "replace")


def _parse_date(name: str) -> str:
    m = DATE_RE.search(name)
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _safe_name(name: str) -> str:
    name = urllib.parse.unquote(name)
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip() or "unnamed.acmi"


def _content_length(url: str, timeout: float) -> int | None:
    try:
        with _request(url, method="HEAD", timeout=timeout) as resp:
            value = resp.headers.get("Content-Length")
            return int(value) if value else None
    except (OSError, urllib.error.URLError, ValueError):
        return None


def _under(base: str, url: str) -> bool:
    return url == base or url.startswith(base)


def crawl(root: str, *, recursive: bool, max_depth: int, timeout: float,
          include: re.Pattern | None, exclude: re.Pattern | None,
          sleep: float, verbose: bool) -> list[Entry]:
    seen: set[str] = set()
    entries: list[Entry] = []

    def walk(url: str, depth: int):
        if depth > max_depth:
            return
        if verbose:
            print(f"[index] {url}", file=sys.stderr)
        try:
            page = _fetch_text(url, timeout)
        except (OSError, urllib.error.URLError) as exc:
            print(f"[warn] 无法读取索引 {url}: {exc}", file=sys.stderr)
            return
        for href in HREF_RE.findall(page):
            href = html.unescape(href).strip()
            if not href or href.startswith(("?", "#", "mailto:")):
                continue
            full = urllib.parse.urljoin(url, href)
            if not _under(root, full):
                continue
            if full in seen:
                continue
            seen.add(full)
            if full.endswith("/"):
                if recursive:
                    walk(full, depth + 1)
                continue
            if not ACMI_RE.search(full):
                continue
            name = _safe_name(Path(urllib.parse.urlparse(full).path).name)
            date = _parse_date(name)
            if include and not include.search(name):
                continue
            if exclude and exclude.search(name):
                continue
            entries.append(Entry(url=full, name=name, date=date))
            if sleep:
                time.sleep(sleep)

    walk(root, 0)
    # 同名文件保留 URL 最短/路径更浅的一个，避免重复下载
    by_name: dict[str, Entry] = {}
    for e in entries:
        old = by_name.get(e.name)
        if old is None or len(e.url) < len(old.url):
            by_name[e.name] = e
    return sorted(by_name.values(), key=lambda e: (e.date, e.name), reverse=True)


def _in_date(date: str, since: str, until: str) -> bool:
    if since and (not date or date < since):
        return False
    if until and (not date or date > until):
        return False
    return True


def _size_ok(size: int | None, min_kb: float, max_mb: float) -> bool:
    if size is None:
        return True
    if min_kb and size < min_kb * 1024:
        return False
    if max_mb and size > max_mb * 1024 * 1024:
        return False
    return True


def cmd_list(args) -> int:
    inc = re.compile(args.include, re.I) if args.include else None
    exc = re.compile(args.exclude, re.I) if args.exclude else None
    entries = crawl(args.root, recursive=not args.no_recursive,
                    max_depth=args.max_depth, timeout=args.timeout,
                    include=inc, exclude=exc, sleep=args.sleep,
                    verbose=args.verbose)
    shown = 0
    for e in entries:
        if not _in_date(e.date, args.since, args.until):
            continue
        size = _content_length(e.url, args.timeout) if args.size else None
        if not _size_ok(size, args.min_size_kb, args.max_size_mb):
            continue
        print(f"{e.date or '-':10s} {'' if size is None else str(size):>12s}  {e.name}")
        print(f"           {e.url}")
        shown += 1
        if args.limit and shown >= args.limit:
            break
    print(f"\n共 {len(entries)} 个 ACMI；本次展示/匹配 {shown} 个。", file=sys.stderr)
    return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(rows: list[dict], dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "manifest.csv"
    fields = ["file", "date", "size", "sha256", "status", "url", "imported_at"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    (dest / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] {csv_path} ({len(rows)} rows)")


def _download_one(entry: Entry, dest: Path, *, timeout: float,
                  retries: int, overwrite: bool,
                  max_bytes: int | None = None) -> dict | None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / entry.name
    if target.exists() and not overwrite:
        size = target.stat().st_size
        print(f"[skip] {target.name} ({size} bytes)")
        return {"file": target.name, "date": entry.date, "size": size,
                "sha256": _sha256(target), "status": "existing",
                "url": entry.url, "imported_at": _now()}
    part = dest / (entry.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with _request(entry.url, timeout=timeout) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) if total else None
                with part.open("wb") as out:
                    read = 0
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        read += len(chunk)
                        if max_bytes and read > max_bytes:
                            raise IOError(f"超过大小上限 {max_bytes} bytes")
                        if total and read > total:
                            raise IOError("下载长度超过 Content-Length")
            if total and part.stat().st_size != total:
                raise IOError(f"长度不匹配：{part.stat().st_size} != {total}")
            os.replace(part, target)
            size = target.stat().st_size
            print(f"[ok] {target.name} ({size} bytes)")
            return {"file": target.name, "date": entry.date, "size": size,
                    "sha256": _sha256(target), "status": "downloaded",
                    "url": entry.url, "imported_at": _now()}
        except (OSError, urllib.error.URLError) as exc:
            print(f"[retry {attempt}/{retries}] {entry.name}: {exc}", file=sys.stderr)
            time.sleep(min(2 ** attempt, 10))
    part.unlink(missing_ok=True)
    return None


def cmd_download(args) -> int:
    inc = re.compile(args.include, re.I) if args.include else None
    exc = re.compile(args.exclude, re.I) if args.exclude else None
    entries = crawl(args.root, recursive=not args.no_recursive,
                    max_depth=args.max_depth, timeout=args.timeout,
                    include=inc, exclude=exc, sleep=args.sleep,
                    verbose=args.verbose)
    dest = Path(args.dest).expanduser().resolve()
    rows: list[dict] = []
    count = 0
    for e in entries:
        if not _in_date(e.date, args.since, args.until):
            continue
        size = _content_length(e.url, args.timeout)
        if not _size_ok(size, args.min_size_kb, args.max_size_mb):
            print(f"[skip-size] {e.name} ({size})")
            continue
        if args.limit and count >= args.limit:
            break
        if args.dry_run:
            print(f"[dry-run] {e.date or '-'} {size or '?':>12} {e.name}")
            count += 1
            continue
        max_bytes = int(args.max_size_mb * 1024 * 1024) if args.max_size_mb else None
        row = _download_one(e, dest, timeout=args.timeout,
                            retries=args.retries, overwrite=args.overwrite,
                            max_bytes=max_bytes)
        if row:
            rows.append(row)
            count += 1
        if args.sleep:
            time.sleep(args.sleep)
    if not args.dry_run:
        # 合并已有 manifest（同一目录）
        old_rows = []
        old_manifest = dest / "manifest.json"
        if old_manifest.is_file():
            try:
                old_rows = json.loads(old_manifest.read_text(encoding="utf-8"))
            except ValueError:
                old_rows = []
        merged = {r["file"]: r for r in old_rows if isinstance(r, dict)}
        merged.update({r["file"]: r for r in rows})
        _write_manifest(list(merged.values()), dest)
    print(f"\n完成：{count} 个文件。目标目录：{dest}")
    return 0


def cmd_import(args) -> int:
    dest = Path(args.dest).expanduser().resolve()
    sources = [Path(p).expanduser().resolve() for p in args.sources]
    files: list[Path] = []
    for src in sources:
        if src.is_file() and src.suffix.lower() == ".acmi":
            files.append(src)
        elif src.is_dir():
            files.extend(p for p in src.rglob("*") if p.is_file()
                         and p.suffix.lower() == ".acmi")
    if not files:
        print("未找到 .acmi 文件", file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sorted(files):
        target = dest / _safe_name(src.name)
        if src.resolve() != target.resolve():
            if target.exists() and not args.overwrite:
                print(f"[skip] {target.name}")
            elif args.move:
                shutil.move(str(src), str(target))
            else:
                shutil.copy2(src, target)
        size = target.stat().st_size
        rows.append({"file": target.name, "date": _parse_date(target.name),
                     "size": size, "sha256": _sha256(target),
                     "status": "imported", "url": "",
                     "imported_at": _now()})
        print(f"[import] {target.name} ({size} bytes)")
    old_manifest = dest / "manifest.json"
    old_rows = []
    if old_manifest.is_file():
        try:
            old_rows = json.loads(old_manifest.read_text(encoding="utf-8"))
        except ValueError:
            old_rows = []
    merged = {r["file"]: r for r in old_rows if isinstance(r, dict)}
    merged.update({r["file"]: r for r in rows})
    _write_manifest(list(merged.values()), dest)
    return 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DCS 公共 ACMI 下载/整理工具")
    ap.add_argument("command", choices=["list", "download", "import"])
    ap.add_argument("--root", default=ROOT_URL)
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--include", default="", help="文件名包含的正则（不区分大小写）")
    ap.add_argument("--exclude", default="", help="文件名排除的正则")
    ap.add_argument("--since", default="", help="YYYY-MM-DD")
    ap.add_argument("--until", default="", help="YYYY-MM-DD")
    ap.add_argument("--min-size-kb", type=float, default=10.0)
    ap.add_argument("--max-size-mb", type=float, default=200.0)
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 个；0=全部")
    ap.add_argument("--dry-run", action="store_true", help="download 时只列出")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--size", action="store_true", help="list 时 HEAD 查询大小")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--move", action="store_true", help="import 时移动而非复制")
    ap.add_argument("sources", nargs="*", help="import 的源目录/文件")
    args = ap.parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "download":
        return cmd_download(args)
    if not args.sources:
        ap.error("import 需要至少一个源目录/文件")
    return cmd_import(args)


if __name__ == "__main__":
    raise SystemExit(main())
