#!/usr/bin/env python3
"""Quick utility to preview a converted EDF with MNE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import mne

DOUBLE_BANANA_PAIRS = [
    # Left longitudinal chain
    ("Fp1", "F7"),
    ("F7", "T7"),
    ("T7", "P7"),
    ("P7", "O1"),
    # Right longitudinal chain
    ("Fp2", "F8"),
    ("F8", "T8"),
    ("T8", "P8"),
    ("P8", "O2"),
    # Left parasagittal chain
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    # Right parasagittal chain
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
    # Midline
    ("Fz", "Cz"),
    ("Cz", "Pz"),
]


def _build_double_banana(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    import mne

    available = set(ch.upper() for ch in raw.ch_names)
    pairs = [(a, b) for a, b in DOUBLE_BANANA_PAIRS if a.upper() in available and b.upper() in available]
    if not pairs:
        return raw.copy()
    new_names = [" - ".join(pair) for pair in pairs]
    bipolar = mne.set_bipolar_reference(
        raw,
        anode=[a for a, _ in pairs],
        cathode=[c for _, c in pairs],
        ch_name=new_names,
        drop_refs=False,
        copy=True,
    )
    return bipolar.pick(new_names)


def _show_or_snapshot(raw: mne.io.BaseRaw, title: str, snapshot: Path | None) -> None:
    show_viewer = snapshot is None
    browser = raw.plot(
        duration=12.0,
        n_channels=len(raw.ch_names),
        scalings=dict(eeg=100e-6),
        block=show_viewer,
        show=show_viewer,
        title=title,
    )
    if snapshot:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        figure = getattr(browser, "figure", None)
        if figure is None:
            raise RuntimeError("MNE viewer did not expose a matplotlib figure for snapshotting")
        figure.savefig(snapshot)
        if hasattr(browser, "close"):
            browser.close()
        print(f"Saved viewer snapshot to {snapshot}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect a converted EDF file with MNE.")
    parser.add_argument(
        "edf_path",
        nargs="?",
        default="out_edf/Patient1_MTU200736UUS_t1.edf",
        help="Path to EDF file (default: %(default)s)",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Optional PNG file to save the montage (useful on headless systems)",
    )
    parser.add_argument(
        "--notch",
        type=float,
        default=50.0,
        help="Notch filter frequency in Hz (default: %(default)s; set 0 to disable)",
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=0.5,
        help="High-pass filter cutoff in Hz (default: %(default)s; set 0 to disable)",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=35.0,
        help="Low-pass filter cutoff in Hz (default: %(default)s; set 0 to disable)",
    )
    args = parser.parse_args(argv)

    try:
        import mne
    except ImportError as exc:
        raise SystemExit("Install 'mne' to use this viewer (e.g. `python -m pip install mne`)") from exc

    edf_path = Path(args.edf_path)
    if not edf_path.exists():
        raise SystemExit(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    print(raw)

    eeg = raw.copy().pick(picks="eeg")
    try:
        eeg.set_montage("standard_1020", match_case=False, on_missing="ignore")
    except RuntimeError:
        pass

    # Apply bandpass filter with configurable cutoffs
    # Use None for disabled bounds (0 means disable)
    low = args.lowcut if args.lowcut and args.lowcut > 0 else None
    high = args.highcut if args.highcut and args.highcut > 0 else None
    if low is not None or high is not None:
        eeg.filter(low, high, fir_design="firwin", verbose=False)

    if args.notch and args.notch > 0:
        eeg.notch_filter(float(args.notch), verbose=False)

    montage_view = _build_double_banana(eeg)

    _show_or_snapshot(montage_view, f"Double-banana view: {edf_path.name}", args.snapshot)


if __name__ == "__main__":
    main()
