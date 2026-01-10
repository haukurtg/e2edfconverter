# Changelog

## Unreleased

-
## 0.2.2 (2026-01-10)

- Resampling: `--resample-to` uses `scipy.signal.resample_poly` (polyphase FIR) and requires scipy.

## 0.2.1 (2026-01-08)

- Add `--split-by-segment` and `--vendor-style`.
- Improve UTF-16 label scanning and event label handling.
- Docs updates (including `.eeg` support status).

## 0.2.0 (2026-01-07)

- Legacy `.eeg` support (experimental).
- Mixed-rate handling via `--resample-to` (including segment-aware resampling).
- Better parsing for segments, channel on/off handling, and EEG offset support.
- EDF+ writer improvements + stricter validation (PyEDFlib).
- JSON sidecar improvements.

## 0.1.1 (2026-01-05)

- Packaging and CLI polish (quick start, CLI options).

## 0.1.0 (2025-12)

- Initial `.e` → EDF converter with EDF+ annotations.
- Optional TUI, filtering, and JSON sidecar support.
