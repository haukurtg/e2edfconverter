# Nicolet `.e`/`.eeg` → EDF

<img src="docs/logo.png" alt="Logo" width="200">

A Python tool to convert Nicolet/Nervus `.e` (the older `.eeg` is a work in progress!) EEG files into standard EDF+ format. No vendor DLLs, no MATLAB (which costs money!), just Python! I couldn't find a native Python way to get `.e` files out of their vendor format, so me and Opus 4.5 wrote this.

> **Acknowledgment**: This project wouldn't exist without the excellent [FieldTrip](https://github.com/fieldtrip/fieldtrip) toolbox. Their MATLAB implementation of the Nervus/Nicolet file format (`read_nervus_header.m` and `read_nervus_data.m`) was the foundation for this Python port. Since then, we've added substantial GUID/event parsing through our own reverse‑engineering work. Thank you to the FieldTrip team!

> **Note**: Some of our reverse‑engineered event labels are (unfortunately) in Norwegian.

## Quick Start

The easiest way — no installation needed:

```bash
# Install uv if you don't have it (https://docs.astral.sh/uv/)
brew install uv  # or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Convert a single file
uv run --isolated nicolet-e2edf --in /path/to/recording.e --out ./edf_output

# Convert a folder of .e/.eeg files
uv run --isolated nicolet-e2edf --in ./my_eeg_folder --out ./edf_output
```

### Interactive Mode

For a guided experience with menus and progress bars:

```bash
uv run --isolated --with rich nicolet-e2edf --ui
```

![TUI Screenshot](docs/tui_screenshot.png)

## CLI Options

| Option | Description |
|--------|-------------|
| `--in` | Input `.e`/`.eeg` file or folder |
| `--out` | Output directory for EDF files |
| `--glob` | Filter pattern when input is a folder (e.g. `Patient1_*`) |
| `--json-sidecar` | Also emit a `.json` with metadata (channels, events, etc.) |
| `--split-by-segment` | Output one EDF per segment if the recording contains multiple segments |
| `--vendor-style` | Suppress system events to better match vendor EDF exports |
| `--resample-to` | Resample to a specific rate (Hz) |
| `--lowcut` | High-pass filter cutoff in Hz (requires scipy) |
| `--highcut` | Low-pass filter cutoff in Hz (requires scipy) |
| `--notch` | Notch filter for powerline noise, e.g. `50` or `60` Hz (requires scipy) |
| `--ui` | Launch interactive terminal UI (requires rich) |
| `--verbose` | Show detailed logging |

**Filtering example:**

```bash
# Clinical defaults: 0.5–35 Hz bandpass + 50 Hz notch
uv run --isolated --with scipy nicolet-e2edf \
    --in ./data --out ./edf_output \
    --lowcut 0.5 --highcut 35 --notch 50
```

**Vendor-style comparison example:**

```bash
# Match vendor-style exports (split per segment + suppress system events)
uv run --isolated nicolet-e2edf \
    --in /path/to/recording.e --out ./edf_output \
    --split-by-segment --vendor-style --json-sidecar
```

## Viewing the Results

There's a bundled viewer script that shows your EDF in a double-banana montage:

```bash
uv run --isolated --with mne python inspect_edf.py ./edf_output/Patient1.edf
```

**Note:** When using the interactive TUI (`--ui`), the viewer is automatically launched with MNE in an isolated environment if needed. No manual installation required!

Options: `--lowcut`, `--highcut`, `--notch`, `--snapshot out.png` (for headless systems).

## Recent changes

- `--split-by-segment` to export one EDF per segment when recordings contain multiple segments
- `--vendor-style` to suppress system events for closer vendor-export parity
- Improved EVENTTYPEINFOGUID label recovery (avoids garbled Unicode labels)
- UTF-8 TAL encoding for EDF+ annotations (preserves non-ASCII text)
- Safer event label handling to avoid overwriting known GUID labels

## Limitations

- Mixed sampling rates: the dominant rate is kept unless `--resample-to` is used
- Events are written as EDF+ annotations
- EVENTTYPEINFOGUID decoding is reverse‑engineered/heuristic (not a vendor‑spec parser)
- Legacy `.eeg` support is experimental: some files convert, but signal data and channel labels can be unreliable.
- Some `.e` recordings store only numeric channel IDs (e.g., `1..64`). In those cases even vendor EDF exports keep numeric labels, so this is expected unless an external montage mapping is available. We still need a consistent way to map these when the source provides it.

## Contributing

Contributions are welcome! If you're working on the EDF writer or want to understand the file format:

- **EDF+ Specification**: A copy of the full EDF+ specification is included at [`docs/EDF+ specification.pdf`](docs/EDF+%20specification.pdf). The official spec is also available at [edfplus.info](https://www.edfplus.info/specs/edfplus.html).
- **Tests**: Run `uv run pytest` to verify EDF+ compliance. We use PyEDFlib as a strict validator.

## License

GPL-3.0 — see `LICENSE`.

This project adapts logic from the [FieldTrip](https://github.com/fieldtrip/fieldtrip) toolbox (GPL-3.0).
