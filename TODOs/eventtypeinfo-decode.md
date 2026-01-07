# Task: Decode EVENTTYPEINFOGUID deterministically

Goal: replace the current heuristic string-scan with a real parser for the Event Type dictionary stored in EVENTTYPEINFOGUID (static + dynamic). This should improve event label coverage beyond hardcoded GUIDs.

## Why
- Event packets only carry a GUID + a label/status field; many GUIDs are unknown.
- EVENTTYPEINFOGUID appears to contain the authoritative GUID → label mapping.
- Other parsers don’t implement this, so decoding it would be a clear improvement.

## Deliverables
1) A small tool/script that dumps EVENTTYPEINFOGUID contents from .e files (static + dynamic packets):
   - raw hex
   - UTF‑16 string extraction with offsets
   - GUID locations
2) A deterministic parser for EVENTTYPEINFOGUID records.
3) Integration into header parsing (replace/augment current heuristic mapping).
4) Tests using a real file (or fixture) to prove increased GUID label resolution.

## Steps
1) Inspect raw EVENTTYPEINFOGUID sections
   - Add a CLI/debug tool (e.g., `nicolet_e2edf/tools/dump_eventtypeinfo.py`) that:
     - locates EVENTTYPEINFOGUID in static packets (and in dynamic packets if present)
     - dumps the raw bytes to a file
     - extracts UTF‑16 strings with offsets
     - records nearby GUIDs (mixed‑endian) for correlation
   - Run the tool on a sample corpus and capture output.

2) Reverse‑engineer record layout
   - Use dumped bytes to identify repeated record structure.
   - Likely fields: GUID, record length, label (UTF‑16), maybe category/color/flags.
   - Confirm with multiple files and dynamic packets.

3) Implement parser
   - Add a structured parser (e.g., `_read_event_type_info_structured`) in `nicolet_e2edf/nicolet/header.py`.
   - Prefer deterministic mapping; fall back to current heuristic when parsing fails.

4) Integrate with dynamic packets
   - If EVENTTYPEINFOGUID appears in InfoChangeStream, map by timestamp.
   - Apply per‑segment or per‑event if dictionary changes over time.

5) Tests
   - Add a regression test that verifies: unknown GUID count decreases for a known file.
   - Keep a small fixture or use a real test file in `EEG_test_files`.

## Notes
- Current behavior uses a heuristic scan and is intentionally conservative.
- This task should not break existing parsing if the new parser fails; fallback should remain.
