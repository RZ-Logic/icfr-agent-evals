"""Shared review schema tests: RB-7 header, ids, sentences, word helpers, LF-only CSV."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from pathlib import Path

import pytest

from icfr_agent_evals import review_schema as rs


def test_rb7_columns() -> None:
    assert rs.RB7_COLUMNS == ("row_kind", "unit_id", "run_label", "log_eval_id", "sample_id", "epoch", "scorers",
                              "route_key", "route_class", "reason", "sentence", "verdict", "note", "reviewer",
                              "date")
    assert ",".join(rs.RB7_COLUMNS) == (
        "row_kind,unit_id,run_label,log_eval_id,sample_id,epoch,scorers,route_key,route_class,reason,sentence,"
        "verdict,note,reviewer,date")


def test_constants() -> None:
    assert rs.ROW_KINDS == ("finding", "narrative", "fm_audit")
    assert rs.FINDING_VERDICTS == ("correct", "wrong", "citation_omission", "cannot_settle", "not_reviewed")
    assert rs.FINDING_NOTE_REQUIRED == {"wrong", "cannot_settle"}
    assert rs.NARRATIVE_VERDICTS == ("none_wrong", "wrong")
    assert rs.AUDIT_VERDICTS == ("real_error", "extractor_error")
    assert rs.ERROR_TYPES[-1] == "other" and "missing_required_element" in rs.ERROR_TYPES
    assert rs.DEFAULT_WHOLE_NARRATIVE_SCOPE == "routed" and rs.WHOLE_NARRATIVE_SCOPES == ("routed", "hard", "all")
    assert rs.DEFAULT_FM_EPOCH_ROUTES == "findings" and rs.FM_EPOCH_ROUTE_MODES == ("findings", "skip")
    assert rs.UNIT_PREFIX == {"finding": "F", "narrative": "N", "fm_audit": "A"}
    assert set(rs.UNIT_PREFIX) == set(rs.ROW_KINDS)
    assert rs.RB7_LIST_SEP == " | " and rs.RECORD_LIST_SEP == ";" and rs.WHOLE_NARRATIVE_SENTENCE == "0"
    assert rs.REVIEWER_DIR == "reviewer" and rs.SEALED_DIR == "sealed"
    for path in rs.PACKET_FILES.values():
        assert path == "packet-meta.json" or path.split("/")[0] in (rs.REVIEWER_DIR, rs.SEALED_DIR)
    assert rs.PACKET_FILES["html"] == "reviewer/packet.html"
    assert rs.RECORD_FILES == {"verdicts": "verdicts.csv", "fm_audit": "fm_audit.csv", "sittings": "sittings.csv"}
    assert len(rs.FORBIDDEN_FACT_WORDS) == 61
    assert rs.STATEMENT.startswith("Every verdict in this record was recorded before")


def test_imports_casefile_only() -> None:
    tree = ast.parse(Path(inspect.getfile(rs)).read_text(encoding="utf-8"))
    local = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1}
    assert local == {"casefile"}


def test_no_dash_characters() -> None:
    src = Path(inspect.getfile(rs)).read_text(encoding="utf-8")
    assert "\u2014" not in src and "\u2013" not in src


def test_group_key_canonical() -> None:
    key = rs.group_key(["DEV-1", "code_review: x", {"b": 1, "a": "\u00e9"}, None, True])
    assert key == '["DEV-1","code_review: x",{"a":"\u00e9","b":1},null,true]'
    assert rs.group_key(("a", "b")) == rs.group_key(["a", "b"])


def test_unit_id_format_and_determinism() -> None:
    gk = rs.group_key(["case", "key"])
    uid = rs.unit_id(20261109, "finding", gk)
    assert re.fullmatch(r"F-[0-9a-f]{12}", uid)
    assert uid == rs.unit_id(20261109, "finding", gk)
    expected = hashlib.sha256(f"20261109|finding|{gk}".encode()).hexdigest()[:12]
    assert uid == f"F-{expected}"
    assert rs.unit_id(20261110, "finding", gk) != uid
    assert rs.unit_id(20261109, "narrative", gk).startswith("N-")
    assert rs.unit_id(20261109, "fm_audit", gk).startswith("A-")
    with pytest.raises(KeyError):
        rs.unit_id(1, "unsettleable", gk)


@pytest.mark.parametrize("text, expected", [
    ("The variance is $5,000. It is M2.", ("The variance is $5,000.", "It is M2.")),
    ("  One line\n\nwith  breaks!  Next?\tLast  ", ("One line with breaks!", "Next?", "Last")),
    ("No terminal stop", ("No terminal stop",)),
    ("Ref DP-1 (USD 1.5) cleared. WD-2 is open.", ("Ref DP-1 (USD 1.5) cleared.", "WD-2 is open.")),
    ("", ()),
])
def test_narrative_sentences(text: str, expected: tuple[str, ...]) -> None:
    assert rs.narrative_sentences(text) == expected


def test_narrative_sentences_keep_raw_forms() -> None:
    # dashes and format characters are not folded in the sentence text
    text = "Ledger \u2212$5 \u2010 net. Sur\u00adplus."
    assert rs.narrative_sentences(text) == ("Ledger \u2212$5 \u2010 net.", "Sur\u00adplus.")


def test_normalize_for_match() -> None:
    assert rs.normalize_for_match("  A\u2013B \u200b  C\uff11 ") == "A-B C1"


def test_forbidden_word_hits() -> None:
    assert rs.forbidden_word_hits("Ledger liabilities") == []
    assert rs.forbidden_word_hits("Nearest match, OK? Correct!") == ["correct", "match", "nearest", "ok"]
    assert rs.forbidden_word_hits("matchbox incorrectly okay passage") == []
    assert rs.forbidden_word_hits("a == b, c != d, \u2713 \u2717") == ["!=", "==", "\u2713", "\u2717"]
    assert rs.forbidden_word_hits("DECOY") == ["decoy"]
    assert rs.forbidden_word_hits("same_value is_match") == ["match", "same"]  # an underscore splits words


def test_model_hint_hits() -> None:
    assert rs.model_hint_hits("Claude's view; GPT-4 and gemini") == ["claude", "gemini", "gpt"]
    assert rs.model_hint_hits("Opusculum, sonnets, haikus, lunar") == []
    assert rs.model_hint_hits("mockllm/model") == ["mockllm"]
    assert rs.model_hint_hits("") == []


def _rows() -> list[dict[str, str]]:
    return [
        {"row_kind": "finding", "unit_id": "F-0123456789ab", "route_key": "code_review: hedged_amount",
         "reason": "hedged_amount: about $5,000 | other", "verdict": "wrong",
         "note": 'line one\nline "two", with comma', "reviewer": "SYNTHETIC-TEST", "date": "2026-11-09"},
        {"row_kind": "narrative", "unit_id": "N-0123456789ab", "sentence": "1;3", "reason": "amount;other",
         "verdict": "wrong", "note": "synthetic test row, not a review"},
    ]


def test_write_rb7_csv_lf_only_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.csv"
    rs.write_rb7_csv(path, _rows())
    data = path.read_bytes()
    assert data.count(b"\r") == 0
    assert not data.startswith(b"\xef\xbb\xbf")
    assert data.split(b"\n", 1)[0] == ",".join(rs.RB7_COLUMNS).encode()
    back = rs.read_rb7_csv(path)
    assert len(back) == 2
    for row, src in zip(back, _rows()):
        assert list(row) == list(rs.RB7_COLUMNS)
        for col in rs.RB7_COLUMNS:
            assert row[col] == src.get(col, "")
    assert data.endswith(b"\n")


def test_write_csv_same_bytes_for_same_rows(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    rs.write_rb7_csv(a, _rows())
    rs.write_csv(b, rs.RB7_COLUMNS, _rows())
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes().decode("utf-8") == rs.csv_text(rs.RB7_COLUMNS, _rows())


def test_write_refuses_carriage_return_and_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        rs.write_rb7_csv(tmp_path / "x.csv", [{"note": "a\r\nb"}])
    with pytest.raises(ValueError):
        rs.write_rb7_csv(tmp_path / "y.csv", [{"not_a_column": "x"}])


def _header() -> bytes:
    return ",".join(rs.RB7_COLUMNS).encode()


def test_read_rb7_refuses_crlf_header(tmp_path: Path) -> None:
    path = tmp_path / "crlf.csv"
    path.write_bytes(_header() + b"\r\n")
    with pytest.raises(ValueError):
        rs.read_rb7_csv(path)


def test_read_rb7_refuses_reordered_header(tmp_path: Path) -> None:
    cols = list(rs.RB7_COLUMNS)
    cols[0], cols[1] = cols[1], cols[0]
    path = tmp_path / "order.csv"
    path.write_bytes(",".join(cols).encode() + b"\n")
    with pytest.raises(ValueError):
        rs.read_rb7_csv(path)


def test_read_rb7_refuses_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + _header() + b"\n")
    with pytest.raises(ValueError):
        rs.read_rb7_csv(path)


def test_read_refuses_cr_in_body_and_short_rows(tmp_path: Path) -> None:
    path = tmp_path / "body.csv"
    path.write_bytes(_header() + b"\n" + b"finding" + b"," * 14 + b"\r\n")
    with pytest.raises(ValueError):
        rs.read_rb7_csv(path)
    path.write_bytes(_header() + b"\nfinding,x\n")
    with pytest.raises(ValueError):
        rs.read_rb7_csv(path)
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        rs.read_rb7_csv(path)


def test_read_csv_any_header(tmp_path: Path) -> None:
    path = tmp_path / "cases.csv"
    rows = [{"case_id": "dev-1", "case_input": "multi\nline", "case_facts_json": json.dumps([{"a": 1}])}]
    rs.write_csv(path, rs.CASE_COLUMNS, rows)
    assert rs.read_csv(path, rs.CASE_COLUMNS) == rows
    with pytest.raises(ValueError):
        rs.read_csv(path, rs.PACKET_COLUMNS)


def test_write_lf_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    rs.write_lf(path, "a\nb\n")
    assert path.read_bytes() == b"a\nb\n"
    assert rs.sha256_file(path) == hashlib.sha256(b"a\nb\n").hexdigest()
