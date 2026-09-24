"""Bio stage 5 (build records): turn the human-reviewed OCR into the structured per-block
records that :mod:`src.bio.s5_link` consumes.

The QA reviewer (``scripts/qa/s4_ocr.py``) writes two review layers per book:

  * ``data/{book}_bio_verified.json`` -- ``{block_id -> verified transcription}``, one
    newline-joined text per person block (line/piece order = RTL: piece 0 = the father
    header ``子<order><name>``, piece 1 = the person's name, then body columns).
  * ``data/{book}_bio_fields.json`` -- ``{block_id -> {piece_idx: role}}`` where role is
    ``father|name|son|none``. This holds ONLY the blocks whose auto-detected roles the
    reviewer OVERRODE; every other block uses :func:`scripts.qa.s4_ocr.detect_fields`
    (father = piece 0, name = piece 1, sons = pieces after a ``生子`` marker).

This builder resolves each block's roles (auto-detect, then apply overrides), pulls the
verified text at those pieces, and emits ``{id, name, father_char, sons, ...}`` in the exact
shape ``s5_link`` reads from ``4_ocr/*.jsonl`` -- so ``s5_link`` is unchanged; it just reads
this reviewed source instead. Output goes to ``books/{book}/bio/5_records/{section}.jsonl``.

Field handling reflects the reviewer's notes on Book 3:

  * **father / name are thoroughly reviewed, low variance** -- trusted as-is. ``father_char``
    is parsed by :func:`src.bio.s4_ocr.parse_father`, which already tolerates a 2-char
    ``子X`` header (no birth-order glyph) as well as ``子<order>X``.
  * **son columns are messier.** Some are pure death/status notes with no name (``长夭``,
    ``次子夭``) -> dropped (they never appear in the graph, so keeping them would break the
    exact-ordered stitch match). Some carry a name plus descriptive text (``宪棕 张出``,
    ``观音夭``) -> the given name is substring-extracted by anchoring on the book's
    generation character. The surviving son NAMES are emitted compacted, in printed (age)
    order; the raw column texts are kept in ``sons_raw`` so nothing is lost.

Run::

    PYTHONPATH=. python -m src.bio.s5_build_bio --book book3
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import os
import re

from src.bio.s4_ocr import parse_father

logger = logging.getLogger(__name__)

# Generation characters that head a formal given name in Books 3 & 4 (given name =
# gen-char + 1). Used to locate the name inside an ordinal-prefixed or annotated column
# (三庆坤 -> 庆坤, 宪棕 张出 -> 宪棕). NOT used to gate whether a column is a name: many real
# sons are 2-char names outside this set (和平, 友明, 国辉 in later/modern branches).
GEN_CHARS = set("传纪广昭宪庆繁祥令")
# The same characters in generation order: a person whose name starts with GEN_ORDER[k] has
# sons named GEN_ORDER[k+1]+X. Used to restore the implicit generation char when a bio lists
# sons by bare given char ("杆 楷" under 昭鸿 = 宪杆, 宪楷).
GEN_ORDER = "传纪广昭宪庆繁祥令"


def son_gen_char(name: str | None) -> str | None:
    """The generation char of this person's sons, or None if their name isn't gen-headed."""
    if not name or name[0] not in GEN_ORDER:
        return None
    k = GEN_ORDER.index(name[0])
    return GEN_ORDER[k + 1] if k + 1 < len(GEN_ORDER) else None

# Birth-order / ordinal glyphs that can PREFIX a son name in a column (长庆煌, 三昭汉, 次昭洪).
ORDINAL_PREFIX = set("长次幼之一二三四五六七八九十")

# Markers that make a column a STATUS note, not a name: died-young / line-terminated /
# adoption-and-inheritance phrases with no own name to match against the graph.
_STATUS = re.compile(r"夭|殇|殀|折|殁|歿|俱止|承嗣|出继|入继|名下|编继|为[广昭宪庆繁祥纪传]")

# Kinship/inheritance phrases that make the WHOLE column a note, not a name -- even the head
# chars before them aren't a person (``双桃承嗣`` = inherits for two branches, not a son 双桃).
# Only applied when no generation char anchors a real name in the column.
_KINSHIP_NOTE = re.compile(r"承嗣|名下|俱止|出继|入继|编继")

# Auto-detect of son columns: the sons follow a "生子<count>名" HEADER -- specifically "生子"
# + a Chinese numeral, NOT any "生子" substring. A birth clause reads "生子道光辛丑…" (子 = the
# 子 year) or "生子民国…"; a bare "生子" match wrongly takes that as a sons header, fabricating a
# son from a date fragment AND skipping the real header later in the block.
#
# Two failure modes bound the son run, so NEITHER the printed count NOR the terminator alone
# is enough:
#   * OVER-read: after the real sons a block may run into free prose that no terminator stops
#     (273_279_0_0: "生子二名" 广镛 广⿰钅席 then a house-building note) -> the count stops it.
#   * UNDER-read: the printed count is sometimes low (273_279_4_7: "生子一名" but lists 繁隆 AND
#     繁荣, then 生女) -> a hard count-gate would drop the real 繁荣.
# So: read candidate columns up to a hard terminator (生女 / bare 女… / life-event), then keep
# the printed count -- plus ONE extra column only when it is the last one before a terminator
# and looks like a name (the off-by-one case). No count -> keep all up to the terminator.
_COUNT = "一二三四五六七八九十两"
# "生于<num>名" is a known misprint of the header (b4 187_189_0_0); a birth clause is always
# "生于<date>", never "生于<num>名", so accepting it is unambiguous.
_SONS_HEADER = re.compile(rf"生子([{_COUNT}])|生于([{_COUNT}])名")
_SONS_END = re.compile(r"^[配继殁歿葬享寿卒女]")  # life-event / 女=daughter column ends the sons
_COUNT_VAL = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _name_like(s: str) -> bool:
    s = s.strip()
    return 2 <= len(s) <= 3 and not _SONS_END.match(s) and "生" not in s


def auto_son_idxs(lines: list[str]) -> list[int]:
    # anchor on a real 生子<count> header (last one wins if a birth clause preceded it).
    start = n_expected = None
    for i, c in enumerate(lines):
        m = _SONS_HEADER.search(c)
        if m:
            start, n_expected = i, _COUNT_VAL.get(m.group(1) or m.group(2), 0)
    if start is None:
        return []
    # candidate son columns run until a hard terminator.
    cand = []
    for i in range(start + 1, len(lines)):
        c = lines[i]
        if re.search("生女", c) or _SONS_END.match(c):
            break
        if c.strip():
            cand.append(i)
    if not n_expected or len(cand) <= n_expected:
        return cand
    # more columns than the printed count: keep the count, plus one extra only if it is the
    # last column before a terminator and looks like a name (tolerates an off-by-one count).
    if len(cand) == n_expected + 1:
        nxt = cand[-1] + 1
        terminated = (nxt >= len(lines) or not lines[nxt].strip()
                      or re.search("生女", lines[nxt]) or _SONS_END.match(lines[nxt]))
        if terminated and _name_like(lines[cand[-1]]):
            return cand
    return cand[:n_expected]


def _lines(text: str) -> list[str]:
    return (text or "").split("\n")


def son_idxs(bid: str, lines: list[str], fields: dict) -> list[int]:
    """Son piece-indices: auto-detect, with the reviewer's overrides applied PER SLOT.

    Same resolution as the QA tool (``effectiveFields`` in scripts/qa/s4_ocr.py): a slot the
    reviewer labelled wins (``son`` adds it, any other role -- incl. ``none`` -- removes it);
    every unlabelled slot keeps its auto-detected role. An override that only unmarks a few
    trailing columns therefore keeps the real sons instead of zeroing them.
    """
    idxs = set(auto_son_idxs(lines))
    for k, role in (fields.get(bid) or {}).items():
        if role == "son":
            idxs.add(int(k))
        else:
            idxs.discard(int(k))
    return sorted(idxs)


def role_idx(bid: str, lines: list[str], fields: dict, role: str, default: int | None) -> int | None:
    """The single piece-index for ``father``/``name``: override if present, else default."""
    override = fields.get(bid) or {}
    hit = [int(k) for k, v in override.items() if v == role]
    if hit:
        return hit[0]
    if default is not None and str(default) in override:
        return None  # the reviewer relabelled the default slot as something else
    return default if default is not None and default < len(lines) else None


def extract_son_name(col: str) -> str | None:
    """Pull the given name out of a son column, or None if it carries no matchable name.

    The 677 clean 2-char columns are the common case; this handles the messy tail without
    guessing (a WRONG son name breaks the exact-ordered stitch match worse than a missing
    one, so ambiguous columns return None to be reported rather than fabricated). Order:

      1. rare-glyph IDS name (``广⿰钅舀`` -- ``⿰`` describes one character) -> keep whole;
      2. drop spaces, then anchor on a generation char if EXACTLY one is present
         (``三庆坤`` -> ``庆坤``, ``宪棕张出`` -> ``宪棕``); two+ gen chars = multiple names -> None;
      3. no gen char: strip a leading ordinal (长/次/三…) and a leading ``子``, then cut at
         the first status marker; keep the first 2 chars if 2 remain and they aren't status
         (``观音夭`` -> ``观音``, ``和平`` -> ``和平``; ``次子夭`` -> None, ``双桃承嗣`` -> None).
    """
    col = col.strip()
    if not col:
        return None
    if "⿰" in col or "⿱" in col or "⿴" in col:      # IDS rare-glyph name, kept whole
        return col
    col = col.replace(" ", "").replace("　", "")

    # A kinship/inheritance note is not a son even when it names a gen-char target: "半昭汉
    # 名下承嗣" = "inherits under 昭汉's name", 昭汉 is the target, not this person's son. So
    # veto BEFORE the gen-char extraction. Likewise "门女子"/"石女子" are stillbirth/daughter
    # placeholders (女子), not sons.
    if _KINSHIP_NOTE.search(col) or "女子" in col:
        return None

    gen_positions = [i for i, ch in enumerate(col) if ch in GEN_CHARS and i + 1 < len(col)]
    if len(gen_positions) == 1:
        i = gen_positions[0]
        # gen char + ORDINAL + status is a death note, not a name ("宪次夭" = 2nd son died young).
        if col[i + 1] in ORDINAL_PREFIX and _STATUS.search(col[i + 1:]):
            return None
        return col[i:i + 2]
    if len(gen_positions) > 1:
        return None  # multiple generation chars => multiple names in one column; don't guess.

    body = col
    while body and (body[0] in ORDINAL_PREFIX or body[0] == "子"):
        body = body[1:]
    if _STATUS.search(body):                          # a status tail: keep only the head
        body = _STATUS.split(body)[0]
    body = body.strip()
    if len(body) < 2:                                 # nothing but ordinal/status left
        return None
    return body[:2]


def extract_son_names(col: str, son_gen: str | None) -> list[str]:
    """All son names in one column -- a column can hold several space-separated sons.

    Tokens (split on whitespace) are one of:
      * a bare given char (``杆 楷``, ``椿 次幼殁``): the generation char is implicit, so it's
        restored from the father's generation (``son_gen``) -> ``宪杆``, ``宪楷``, ``宪椿``;
      * a gen-headed name, possibly with a note token beside it (``宪棕 张出``: X出 = born of
        mother X; ``庆有 为昭``: heir to 昭…) -> the name, via :func:`extract_son_name`;
      * a note / status token (``夭``, ``次幼殁``, ``张出``, ``二三``) -> skipped.
    A single-token column goes straight to :func:`extract_son_name`. A lone single-char
    column (``沺``, ``广``, ``下``) is a wrapped name fragment and is NOT expanded (no guess).
    """
    tokens = col.split()
    if len(tokens) <= 1:
        nm = extract_son_name(col)
        return [nm] if nm else []
    out = []
    for tok in tokens:
        if len(tok) == 1:
            if (son_gen and tok not in ORDINAL_PREFIX and tok not in GEN_CHARS
                    and not _STATUS.search(tok)):
                out.append(son_gen + tok)
        elif any(ch in GEN_CHARS for ch in tok):
            nm = extract_son_name(tok)
            if nm:
                out.append(nm)
        # else: a note/status token (张出, 次幼殁, 二三) -- skip
    return out


def build_record(bid: str, text: str, fields: dict) -> dict:
    lines = _lines(text)
    qa_flags = []
    override = fields.get(bid) or {}
    # A "son" label on the NAME slot (1) can't be right -- son columns only follow the 生子
    # header -- it's a mis-click (b4 102_102_1_0: 三长 labelled son, but the tree has 三长 with
    # the bio's father). Keep slot 1 as the name and flag it for review; the data file is
    # the reviewer's and is never rewritten here.
    name_slot_conflict = override.get("1") == "son" and not any(
        _SONS_HEADER.search(c) for c in lines[:1])
    if name_slot_conflict:
        fields = {**fields, bid: {k: v for k, v in override.items() if k != "1"}}
        qa_flags.append("fields_conflict: slot 1 (name) labelled son; kept as name")
    fa_i = role_idx(bid, lines, fields, "father", 0)
    nm_i = role_idx(bid, lines, fields, "name", 1)
    father_header = lines[fa_i] if fa_i is not None else ""
    name = lines[nm_i].strip() if nm_i is not None else None

    sons_raw, sons = [], []
    for i in son_idxs(bid, lines, fields):
        if i >= len(lines) or not lines[i].strip():
            continue
        raw = lines[i].strip()
        sons_raw.append(raw)
        # pure-status (died-young) entries yield nothing; a column may yield several sons.
        sons.extend(extract_son_names(raw, son_gen_char(name)))

    # match the field shape s5_link reads (per-reader dicts; only the verified read here).
    return {
        "id": bid,
        "generation": None,          # filled below from the block sidecar
        "name": {"vision": name},
        "father_char": {"vision": parse_father([father_header]) if father_header else None},
        "father_header": father_header,   # full 子<order><name>, kept for father-char QA
        "sons": {"vision": sons},
        "sons_raw": sons_raw,
        "daughters": {"vision": []},
        "birth": {"vision": None},
        "qa_flags": qa_flags,
        "raw": {"vision": {"text": text}},   # lossless: the full verified transcription
    }


def build_book(book: str, data_dir: str = "data", books_dir: str = "books") -> int:
    verified = json.load(open(os.path.join(data_dir, f"{book}_bio_verified.json")))
    fields_path = os.path.join(data_dir, f"{book}_bio_fields.json")
    fields = json.load(open(fields_path)) if os.path.exists(fields_path) else {}

    # generation comes from the existing 4_ocr records (same block ids), which already
    # carry the band/generation; reuse them so we don't re-derive geometry here.
    gen_of: dict[str, int] = {}
    for f in glob.glob(os.path.join(books_dir, book, "bio", "4_ocr", "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            gen_of[r["id"]] = r.get("generation")

    out_dir = os.path.join(books_dir, book, "bio", "5_records")
    os.makedirs(out_dir, exist_ok=True)
    by_section: dict[str, list[dict]] = collections.defaultdict(list)
    for bid, text in verified.items():
        rec = build_record(bid, text, fields)
        rec["generation"] = gen_of.get(bid)
        # section = the {start}_{end} stem that heads the block id (e.g. 18_63 from 18_63_2_2)
        section = "_".join(bid.split("_")[:2])
        by_section[section].append(rec)

    n = 0
    for section, recs in by_section.items():
        recs.sort(key=lambda r: [int(x) for x in r["id"].split("_")])
        with open(os.path.join(out_dir, f"{section}.jsonl"), "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1

    missing_gen = sum(1 for recs in by_section.values() for r in recs if r["generation"] is None)
    total_sons = sum(len(r["sons"]["vision"]) for recs in by_section.values() for r in recs)
    logger.info("%s: built %d records across %d sections -> %s (%d sons, %d missing generation)",
                book, n, len(by_section), out_dir, total_sons, missing_gen)
    return n


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--books-dir", default="books")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(message)s")
    build_book(a.book, a.data_dir, a.books_dir)


if __name__ == "__main__":
    main()
