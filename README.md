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

When you leave the default glob in place the converter will also try `.eeg` files in the folder—yes, that old-as-dust format—which usually works unless the file dates back to the EEG stone age.
```

### CLI options at a glance

* `--glob`: alternative discovery pattern when the input path is a directory (default adds `.eeg`).
* `--patient-json`: optional path to a JSON file with per-glob metadata overrides. Each object needs a `"glob"` key plus the EDF patient fields to override, e.g. `[{"glob": "Patient*.e", "PatientName": "Anon Subject"}]`.
* `--json-sidecar`: emit a `<case>.json` alongside the EDF with channel labels, sample count, sampling rate, start time (when present), and events.
* `--resample-to`: resample the output to a single sampling rate before writing.
* `--verbose`: surface converter logging.

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

## Developing

* Format and lint with Ruff/Black: `ruff check .` and `black --check .`
* Run the full test suite: `pytest`

Limitations to keep in mind while hacking: mixed sampling rates inside a single conversion are dropped (only the dominant rate is kept) and the EDF writer currently sticks to EDF+ annotations for events.

## Roadmap

This converter already covers end-to-end `.e` → EDF (plus JSON sidecar) for modern cases. Remaining items are nice-to-haves rather than blockers:

1. Publish de-identified `.e` fixtures and wire GitHub Actions (ruff, black --check, pytest) against them.
2. Offer gentler mixed-rate handling (e.g., optional resampling) as an alternative to today’s dominant-rate approach.
3. Add usage examples that demonstrate metadata overrides and the existing sidecar/event handling.

Reality check: validated on 500 Hz traces, keeps dominant-rate channels when rates diverge, and awaits broader fixture coverage to lock in regression tests.

## Attribution

This converter contains logic adapted from the FieldTrip toolbox (GPL-3.0), in particular `read_nervus_data.m` and `read_nervus_header.m` from the FieldTrip project: https://github.com/fieldtrip/fieldtrip.

## Licence

Released under the GPL-3.0 license; see `LICENSE` for full terms. If you redistribute wheels or docker images, ship the licence and attribution with them—future-you (and the FieldTrip authors) will thank you.
