# Nicolet `.e` → EDF

I realised there was still no native Python way to pry Nicolet/Nervus `.e` files out of their vendor bubble and into something sane like EDF. So this repo is a small collection of scripts that read modern `.e` studies and spit out EDF you can actually feed to MNE, EEGLAB, or whatever analysis rabbit hole you’re in.

No vendor DLLs, no MATLAB runtime, no mystery GUIs—just Python 3.10+, NumPy, and the legally-required GPL boilerplate.

## Quick start

```
python3.10 -m venv .venv        # grab Python ≥3.10 (pyenv / Homebrew) for a smooth install
source .venv/bin/activate        # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -e .[dev]            # pulls pytest, ruff, etc.
```

You now have a `nicolet-e2edf` CLI on your PATH:

```
nicolet-e2edf --in /path/to/case.e --out ./edf_drop
nicolet-e2edf --in ./eeg_files --out ./edf_drop --glob "*.e"
```

The CLI keeps only the dominant sampling-rate channels (mirrors FieldTrip), fakes anonymised IDs when you don’t give it patient metadata, and logs what it threw away. Want deterministic metadata? Point `--patient-json` at a file full of `{"glob": "Patient*.e", "PatientName": "..."}` blocks and it’ll do the substitutions.

## Looking at the result

Once you have an EDF, point the bundled MNE helper at it:

```
.venv/bin/python inspect_edf.py edf_drop/Patient1.edf
```

You’ll get a 0.5–35 Hz / 60 Hz notch, double-banana montage, 12‑second window, 100 µV/cm scaling—basically what the neurologist on call expects. On a headless box, pass `--snapshot out.png` and it will quietly save the figure instead of trying (and failing) to pop a window.

## What’s actually in here

- `nicolet/header.py` – the clean-room reinterpretation of FieldTrip’s header reader, mapping GUIDs, segments, events.
- `nicolet/data.py` – sequential waveform reader that hands you µV traces for chosen channels/ranges.
- `nicolet/edf_writer.py` – minimal EDF+ writer, nothing fancy, just correct scaling and timestamps.
- `nicolet/cli.py` – the `nicolet-e2edf` entry point with batch support + metadata rules.
- `inspect_edf.py` – the helper described above.
- `tools/view_with_mne.py` – quick peek straight from `.e` without converting, if you really need it.
- `tests/` – unit + end-to-end tests with synthetic headers/data so CI doesn’t need real PHI.

## Roadmap

1. Ship de-identified `.e` fixtures and wire GitHub Actions (ruff, black --check, pytest).
2. Handle mixed sampling rates in a less brutal way (resample or emit per-rate EDFs).
3. Surface annotations/events either in EDF+ or a JSON sidecar so they don’t vanish.

Reality check: currently validated on 500 Hz traces, drops off-rate channels, and still needs broader fixture coverage.

## Attribution

This converter contains logic adapted from the FieldTrip toolbox (GPL-3.0), in particular `read_nervus_data.m` and `read_nervus_header.m` from the FieldTrip project: https://github.com/fieldtrip/fieldtrip.

## Licence

Released under the GPL-3.0 license; see `LICENSE` for full terms. If you redistribute wheels or docker images, ship the licence and attribution with them—future-you (and the FieldTrip authors) will thank you.
