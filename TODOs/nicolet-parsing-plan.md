# Nicolet Parsing Plan (GUIDs, Events, Legacy)

This document is a work plan to improve **parsing/reading of Nicolet/Nervus `.e` files** in this repo, with emphasis on:
- Parsing **more GUID-backed structures** (especially events + annotations).
- Improving robustness across real-world Nicolet variants.
- Adding support for **legacy (pre-ca. 2012) Nicolet files** currently rejected.

No work in this file is executed yet; it’s a roadmap we can implement step-by-step later.

---

## 0) Quick Context (Current State)

Primary parsing entrypoints:
- Header: `nicolet_e2edf/nicolet/header.py` (`read_nervus_header`)
- Waveform: `nicolet_e2edf/nicolet/data.py` (`read_nervus_data`)

What is currently parsed:
- Static packet table + QI indices + main index (section offsets/lengths).
- `InfoGuids` list (GUID values only).
- `InfoChangeStream` dynamic packet headers, and raw payload bytes are collected for dynamic packets whose GUID exists as a static packet tag.
- `TSGUID` payload is parsed into `TSInfo` (from dynamic if present, else static).
- `SegmentStream` is parsed into segments (timestamps, duration, per-channel sample counts derived from TSInfo sampling).
- `Events` section is parsed into event markers; annotation text is read only for the specific “Annotation” event GUID.

What is *not* parsed (key gaps):
- The `EVENTTYPEINFOGUID` section appears to contain a rich event dictionary, but it is not parsed; event GUIDs remain mostly `UNKNOWN`.
- Additional event metadata sections like `EventData` and `EventLocation` exist in some files but are ignored.
- Other static/dynamic GUID payloads (patient info, study info, montage info, filters, display, etc.) are generally not decoded into structured data in Python.
- Legacy pre-2012 `.e` layout is explicitly rejected when `indexIdx == 0`.

---

## 1) Goals & Success Criteria

### Parsing completeness (events/annotations)
- Preserve *all* event markers as today (never lose existing output).
- Resolve event GUIDs into human-readable names whenever possible (preferably from file contents, not hardcoded maps).
- Extract annotation text reliably for all event types that carry text, not only the single hardcoded “Annotation” GUID.
- Optionally capture additional event metadata (location, channel association, etc.) if present.

### Legacy support (pre-2012)
- Be able to open and parse legacy `.e` files that currently throw: “Unsupported old-style Nicolet file format (pre-ca. 2012)”.
- Define exactly what “supported” means for legacy:
  - At minimum: read TS/channel definitions, segment boundaries, and waveform.
  - If events exist in legacy layout, parse them too.

### Robustness
- Gracefully handle missing sections and variants:
  - Missing `InfoChangeStream`, missing `TSGUID` dynamic packets, missing `Events`, etc.
- Avoid silent truncation: if a section is partially readable, return what we can and surface a warning/diagnostic signal.

### Tests
- Add tests that verify event GUID resolution and annotation decoding on real-ish samples (from `EEG_test_files`).
- Add tests to prevent regressions in:
  - number of events extracted,
  - number of annotations extracted,
  - timing conversion (OLE/frac → POSIX datetime).

---

## 2) Workstream A: Event Type Dictionary (`EVENTTYPEINFOGUID`)

### A1) Confirm section presence & locate bytes
Tasks:
- In `read_nervus_header`, locate the `StaticPacket` whose `IDStr == "EVENTTYPEINFOGUID"`.
- Use `MainIndex` to read its section bytes (similar to other packets).

Deliverable:
- A small helper that returns the raw buffer for `EVENTTYPEINFOGUID` for inspection and parsing.

### A2) Reverse-engineer the internal format
Tasks:
- Decode `EVENTTYPEINFOGUID` payload into a stable structured representation, ideally:
  - `guidAsStr` (pretty GUID)
  - `name` (human label)
  - flags/fields that indicate if text payload exists, if it’s a system marker, etc. (if discoverable)
- Use a combination of:
  - pattern search for GUID byte sequences,
  - UTF-16LE string blocks,
  - FieldTrip/other reference implementations (if needed later) as guidance.

Deliverable:
- A parser function like `parse_event_type_info(buffer) -> dict[guidAsStr, EventType]`.

### A3) Integrate event type dictionary into `_read_events`
Tasks:
- Build a lookup table: `event_type_by_guid` from `EVENTTYPEINFOGUID`.
- Replace/augment `_EVENT_GUID_LABELS`:
  - Keep `_EVENT_GUID_LABELS` as a fallback for older files that might not contain a readable dictionary.
  - Prefer dictionary-derived labels when available.
- Set `EventItem.IDStr` using the dictionary label when present.

Deliverable:
- Events now appear with meaningful types instead of `UNKNOWN` for many GUIDs.

### A4) Decide how to expose the dictionary in outputs
Options:
- Store on `NervusHeader` (e.g., `EventTypeInfo`) for diagnostics and downstream use.
- Optionally include it in JSON sidecar under a compact key (e.g., `event_types_seen`).

Deliverable:
- A consistent place to view the mapping for debugging.

---

## 3) Workstream B: Events, Annotations, and “Missing” GUIDs

### B1) Make event parsing resilient to packet length variants
Observation:
- Event packet lengths can vary (e.g., 272, 280, larger), so relying on a fixed post-layout can be fragile if fields are optional.

Tasks:
- Treat packet length as authoritative and parse within it using bounds checks.
- For annotation events (and other text-carrying events), ensure:
  - reserved blocks are skipped only if present within bounds,
  - text length is validated against packet remaining bytes,
  - UTF-16 decoding stops at first null char.

Deliverable:
- No crashes on odd packet sizes; no truncation when text exists.

### B2) Determine which events can carry text (beyond “Annotation”)
Tasks:
- Use the event type dictionary (Workstream A) to identify text-capable event types if that metadata exists.
- If the dictionary doesn’t include explicit metadata:
  - Implement a heuristic: if `text_len > 0`, attempt to read text safely for *any* event, within packet bounds.
  - Only attach `annotation` if decoded text is non-empty and looks valid.

Deliverable:
- More complete annotation extraction across event types.

### B3) Parse `EventData` and `EventLocation` when present
Tasks:
- Detect `StaticPacket.tag == "EventData"` and `StaticPacket.tag == "EventLocation"`.
- Read their bytes and determine structure:
  - likely arrays keyed by event index or event ID.
- Join with events:
  - determine the key (event ordering vs. eventID field we currently skip).
  - store enriched metadata per event (e.g., channel index, sample index, montage/trace association).

Deliverable:
- Optional rich per-event metadata for downstream use (JSON sidecar, annotations).

### B4) Guarantee we don’t “lose” event GUIDs
Tasks:
- Add a debug-mode report that logs:
  - number of events read,
  - unique GUID count,
  - number that remained unresolved (`UNKNOWN`).
- Ensure this can be surfaced via CLI `--verbose` without changing default output.

Deliverable:
- Faster diagnosis when users report “missing events”.

---

## 4) Workstream C: Dynamic Packets Beyond TSInfo

Goal:
- Many dynamic packets are currently collected as raw bytes but never interpreted.

Tasks (prioritized):
- Implement minimal metadata extraction for selected GUIDs:
  - `CHANNELGUID`, `DERIVATIONGUID`, `FILTERGUID`, `DISPLAYGUID`
- Identify which of these directly affect waveform interpretation:
  - channel enable/disable,
  - scaling/resolution changes,
  - sampling rate changes,
  - montage derivations.
- Decide how to represent “changes over time”:
  - keep full time-stamped changes,
  - or compute a “best effective” state at segment start.

Deliverable:
- Structured `DynamicPacketsParsed` (or similar) for the most valuable sections.

---

## 5) Workstream D: Legacy (Pre-ca. 2012) `.e` Support

### D1) Define “legacy” signatures
Today legacy is detected by `indexIdx == 0` at file header.

Tasks:
- Collect a small set of real legacy files (or anonymized headers) to validate assumptions.
- Confirm whether “indexIdx==0” always implies legacy, and whether other distinguishing fields exist.

Deliverable:
- Clear detection logic and sample fixtures.

### D2) Implement legacy index reader
Tasks:
- Find how older layouts encode:
  - static packet table (is offset 172 still valid?),
  - section indices / offsets,
  - waveform stream layout.
- Implement a separate code path:
  - e.g., `read_nervus_header_legacy(...)` that returns the same `NervusHeader` shape when possible.
- Keep modern path unchanged; choose path based on detection.

Deliverable:
- Legacy `.e` can be opened and at least TS/segments/waveform are readable.

### D3) Event parsing for legacy
Tasks:
- If legacy has events in a different section/structure:
  - implement a legacy event reader that produces `EventItem` list.
- If legacy doesn’t have comparable events:
  - return empty `Events` and ensure downstream code handles it.

Deliverable:
- Consistent API behavior across file eras.

---

## 6) Workstream E: Validation, Tooling, and Tests

### E1) Golden metrics on sample corpus
For all files in `EEG_test_files/eeg_files/*.e`:
- number of channels (`TSInfo` length)
- number of segments
- number of events
- number of annotations with text
- number of unresolved event GUIDs

Deliverable:
- A small script (or test) that prints a table to compare before/after.

### E2) Tests
Add tests that:
- Ensure we still read at least the previous number of event packets per file.
- Verify that event GUID resolution improves (e.g., fewer `UNKNOWN` types) when the dictionary exists.
- Verify annotation text extraction on known files containing non-ASCII text (UTF-16LE decoding).
- (Later) Add legacy fixtures/tests once legacy support exists.

Deliverable:
- Regression protection for the parsing improvements.

### E3) Performance sanity
Tasks:
- Ensure parsing large files does not load huge sections unnecessarily.
- For large sections (like event type dictionaries), read-once and reuse.

Deliverable:
- No major slowdown in typical conversions.

---

## 7) Execution Order (Recommended)

1. Implement Workstream A (event type dictionary) end-to-end.
2. Implement Workstream B2 (generalized safe text extraction for events).
3. Implement Workstream B3 (EventData/EventLocation parsing) if needed for your clinical use-cases.
4. Expand Workstream C for dynamic packets that materially affect interpretation.
5. Finally tackle Workstream D (legacy pre-2012) once we have sample files and confirm the on-disk structure.

---

## 8) Open Questions (To Answer Before/While Implementing)

- For the “older Nicolet files” you care about: are they *actually* `indexIdx==0` legacy, or simply different packets/offsets within modern indexing?
- Do you need montage derivations to be represented explicitly in EDF (bipolar derivations), or is raw channel data sufficient?
- For events: is the priority to (a) label everything correctly, (b) preserve all raw GUIDs and text, or (c) infer channel/sample locations for each event?

