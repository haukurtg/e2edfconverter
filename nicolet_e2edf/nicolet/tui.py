from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def rich_available() -> bool:
    return importlib.util.find_spec("rich") is not None


def mne_available() -> bool:
    """Check if MNE is available in the current environment."""
    return importlib.util.find_spec("mne") is not None


def _build_uv_command(script_path: Path, *args: str) -> list[str]:
    """
    Build a uv run command that works whether we're already in a uv environment or not.
    Uses --isolated --with mne to install MNE in an isolated, non-persistent environment.
    """
    return ["uv", "run", "--isolated", "--with", "mne", "python", str(script_path)] + list(args)


def _arrow_menu_select(title: str, options: list[str], *, default_index: int = 0) -> int | None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None
    try:
        import curses
    except Exception:
        return None

    if not options:
        return None

    def _run(screen) -> int | None:
        curses.curs_set(0)
        screen.keypad(True)
        idx = max(0, min(default_index, len(options) - 1))

        while True:
            screen.erase()
            height, width = screen.getmaxyx()
            header = f"{title}  (↑/↓ to move, Enter to select, q to cancel)"
            screen.addnstr(0, 0, header, max(width - 1, 0))

            visible_height = max(height - 2, 1)
            start = max(0, idx - visible_height // 2)
            start = min(start, max(len(options) - visible_height, 0))
            end = min(start + visible_height, len(options))

            for row, opt_index in enumerate(range(start, end), start=1):
                prefix = "> " if opt_index == idx else "  "
                text = prefix + options[opt_index]
                if opt_index == idx:
                    screen.attron(curses.A_REVERSE)
                    screen.addnstr(row, 0, text, max(width - 1, 0))
                    screen.attroff(curses.A_REVERSE)
                else:
                    screen.addnstr(row, 0, text, max(width - 1, 0))

            screen.refresh()
            key = screen.getch()
            if key in {curses.KEY_UP, ord("k")}:
                idx = (idx - 1) % len(options)
            elif key in {curses.KEY_DOWN, ord("j")}:
                idx = (idx + 1) % len(options)
            elif key in {curses.KEY_ENTER, 10, 13}:
                return idx
            elif key in {ord("q"), 27}:
                return None

    try:
        return curses.wrapper(_run)
    except Exception:
        return None


@dataclass(frozen=True)
class TuiOptions:
    json_sidecar: bool
    resample_to: float | None
    lowcut: float | None
    highcut: float | None
    notch: float | None
    verbose: bool


class EEGWave:
    def __init__(self, *, width: int | None = None, height: int = 9) -> None:
        self._width = width
        self._height = height

    @staticmethod
    def _braille_from_pixels(pixels: list[list[bool]], x0: int, y0: int) -> str:
        dot = 0
        # 2x4 braille cell
        # (x,y) -> dot bit:
        # (0,0)=1, (0,1)=2, (0,2)=4, (1,0)=8, (1,1)=16, (1,2)=32, (0,3)=64, (1,3)=128
        mapping = {
            (0, 0): 0x01,
            (0, 1): 0x02,
            (0, 2): 0x04,
            (1, 0): 0x08,
            (1, 1): 0x10,
            (1, 2): 0x20,
            (0, 3): 0x40,
            (1, 3): 0x80,
        }
        for dx in (0, 1):
            for dy in (0, 1, 2, 3):
                yy = y0 + dy
                xx = x0 + dx
                if 0 <= yy < len(pixels) and 0 <= xx < len(pixels[0]) and pixels[yy][xx]:
                    dot |= mapping[(dx, dy)]
        return chr(0x2800 + dot)

    @staticmethod
    def _render_braille_wave(*, t: float, width_chars: int, height_chars: int) -> list[str]:
        pixel_w = max(width_chars * 2, 2)
        pixel_h = max(height_chars * 4, 4)
        mid = (pixel_h - 1) / 2.0

        pixels = [[False for _ in range(pixel_w)] for _ in range(pixel_h)]

        # Soft baseline.
        baseline = int(round(mid))
        for x in range(pixel_w):
            pixels[baseline][x] = True

        def smooth_noise(x: float) -> float:
            # Deterministic "noisy" mix without random module.
            return (
                0.55 * math.sin(x)
                + 0.25 * math.sin(x * 2.13 + 1.4)
                + 0.20 * math.sin(x * 3.91 + 0.7)
                + 0.10 * math.sin(x * 7.77 + 2.2)
            )

        # Simulated EEG: small, busy oscillations + occasional sharper transients.
        for xp in range(pixel_w):
            x_norm = xp / (pixel_w - 1)
            phase = t * 3.2 + x_norm * 14.0
            alpha = smooth_noise(phase) * 0.18
            beta = smooth_noise(phase * 1.9) * 0.08
            drift = math.sin(t * 0.35 + x_norm * 2.0) * 0.05
            value = alpha + beta + drift

            # A few brief spikes to sell the trace as electrophysiology.
            spike_phase = (t * 0.55 + x_norm * 3.0) % 1.0
            if spike_phase < 0.03:
                value += (0.03 - spike_phase) * 9.0
            elif spike_phase > 0.97:
                value -= (spike_phase - 0.97) * 9.0

            # Keep it "compressed": small amplitude around baseline.
            y = int(round(mid - value * (pixel_h * 0.9)))
            y = max(0, min(pixel_h - 1, y))

            pixels[y][xp] = True
            # Add a tiny thickness for readability.
            if y + 1 < pixel_h:
                pixels[y + 1][xp] = True

        lines: list[str] = []
        for row in range(height_chars):
            line_chars: list[str] = []
            y0 = row * 4
            for col in range(width_chars):
                x0 = col * 2
                line_chars.append(EEGWave._braille_from_pixels(pixels, x0, y0))
            lines.append("".join(line_chars))
        return lines

    def __rich_console__(self, console, options):  # type: ignore[no-untyped-def]
        from rich.text import Text

        # Use braille for higher resolution in a compact block.
        if self._width is None:
            width_chars = max(10, options.max_width)
        else:
            width_chars = max(10, min(self._width, options.max_width))
        height_chars = self._height
        t = time.time()
        lines = self._render_braille_wave(t=t, width_chars=width_chars, height_chars=height_chars)
        text = Text("\n".join(lines), style="cyan")
        yield text


def run_rich_wizard(*, title: str = "nicolet-e2edf") -> argparse.Namespace:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text

    console = Console()
    console.print(
        Panel(
            Group(Text("Interactive setup", style="bold green"), EEGWave(width=None, height=8)),
            title=title,
        )
    )

    def _candidate_input_dirs() -> list[Path]:
        cwd = Path.cwd()
        candidates: list[Path] = []
        preferred = [
            cwd / "EEG_Files",
            cwd / "EEG_files",
            cwd / "EEG_test_files" / "eeg_files",
            cwd / "EEG_test_files",
        ]
        for path in preferred:
            if path.is_dir():
                candidates.append(path)
        for child in sorted(cwd.iterdir()):
            if child.is_dir() and child.name not in {p.name for p in candidates}:
                candidates.append(child)
        return candidates[:12]

    def _show_dir_menu() -> None:
        dirs = _candidate_input_dirs()
        if not dirs:
            return
        table = Table(title="Detected folders (from current directory)", show_header=True, header_style="bold blue")
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Path", style="yellow")
        for idx, folder in enumerate(dirs, start=1):
            table.add_row(str(idx), str(folder))
        console.print(Panel(table, border_style="blue"))

    mode = Prompt.ask(
        "Convert from",
        choices=["directory", "files"],
        default="directory",
        show_choices=True,
    )

    input_paths: list[str] = []
    glob_pattern = "*.e"
    if mode == "directory":
        _show_dir_menu()
        candidates = _candidate_input_dirs()
        default_dir = str(candidates[0]) if candidates else "."
        chosen_index = _arrow_menu_select("Select input directory", [str(p) for p in candidates], default_index=0)
        if chosen_index is not None:
            input_dir = str(candidates[chosen_index])
        else:
            input_dir = default_dir
        while True:
            input_dir = Prompt.ask("Input directory", default=input_dir)
            if Path(input_dir).expanduser().is_dir():
                break
            console.print(f"[bold red]Folder not found:[/bold red] {input_dir}")
            _show_dir_menu()
        glob_pattern = Prompt.ask("Glob filter (applies to .e/.eeg)", default="**/*")
        input_paths = [input_dir]
    else:
        while True:
            raw = Prompt.ask("Input files (space-separated; wildcards allowed)", default="")
            tokens = [part for part in raw.split() if part]
            if tokens:
                break
            console.print("[bold red]Please enter at least one file path.[/bold red]")
        input_paths = tokens

    output_dir = Prompt.ask("Output directory", default="./edf_drop")
    json_sidecar = Confirm.ask("Write JSON sidecars?", default=False)
    resample_raw = Prompt.ask("Resample output to Hz? (blank for none)", default="", show_default=False)
    resample_to = float(resample_raw) if resample_raw.strip() else None

    # Filter settings (optional - default preserves raw signal)
    lowcut_raw = Prompt.ask(
        "High-pass filter cutoff Hz? (blank for none, e.g. 0.5)",
        default="",
        show_default=False,
    )
    lowcut = float(lowcut_raw) if lowcut_raw.strip() else None
    highcut_raw = Prompt.ask(
        "Low-pass filter cutoff Hz? (blank for none, e.g. 35)",
        default="",
        show_default=False,
    )
    highcut = float(highcut_raw) if highcut_raw.strip() else None
    notch_raw = Prompt.ask(
        "Notch filter Hz? (blank for none, e.g. 50 or 60 for powerline)",
        default="",
        show_default=False,
    )
    notch = float(notch_raw) if notch_raw.strip() else None

    patient_json = Prompt.ask("Patient rules JSON (blank for none)", default="", show_default=False)
    verbose = Confirm.ask("Verbose logging?", default=False)
    use_tui = Confirm.ask("Use animated progress UI during conversion?", default=True)

    # Build a readable filter description
    if lowcut is not None and highcut is not None:
        filter_desc = f"{lowcut}–{highcut} Hz (bandpass)"
    elif lowcut is not None:
        filter_desc = f">{lowcut} Hz (high-pass)"
    elif highcut is not None:
        filter_desc = f"<{highcut} Hz (low-pass)"
    else:
        filter_desc = "(none)"
    
    notch_desc = f"{notch} Hz" if notch else "(none)"

    table = Table(title="Summary", show_header=False, box=None)
    table.add_row("inputs", " ".join(input_paths) if input_paths else "(none)")
    table.add_row("glob", glob_pattern if mode == "directory" else "(n/a)")
    table.add_row("out", output_dir)
    table.add_row("json sidecar", str(json_sidecar))
    table.add_row("resample_to", str(resample_to) if resample_to else "(none)")
    table.add_row("bandpass", filter_desc)
    table.add_row("notch", notch_desc)
    table.add_row("patient_json", patient_json or "(none)")
    table.add_row("verbose", str(verbose))
    table.add_row("ui", "tui" if use_tui else "plain")
    console.print(Panel(table, border_style="blue"))

    if not Confirm.ask("Proceed?", default=True):
        raise SystemExit(1)

    return argparse.Namespace(
        input_paths=input_paths,
        output_dir=output_dir,
        glob=glob_pattern,
        patient_json=Path(patient_json) if patient_json else None,
        json_sidecar=json_sidecar,
        resample_to=resample_to,
        lowcut=lowcut,
        highcut=highcut,
        notch=notch,
        verbose=verbose,
        tui=use_tui,
        wiz=False,
        ui=True,
    )


def _read_edf_summary(path: Path) -> dict[str, str]:
    data = path.read_bytes()
    if len(data) < 256:
        return {"path": str(path), "error": "too small to be EDF"}
    header_bytes = int(data[184:192].decode("ascii", errors="ignore").strip() or "0")
    n_records = int(data[236:244].decode("ascii", errors="ignore").strip() or "0")
    record_duration = data[244:252].decode("ascii", errors="ignore").strip()
    n_signals = int(data[252:256].decode("ascii", errors="ignore").strip() or "0")
    labels_blob = data[256 : 256 + 16 * n_signals]
    labels = [
        labels_blob[i * 16 : (i + 1) * 16].decode("ascii", errors="ignore").strip()
        for i in range(n_signals)
    ]
    return {
        "path": str(path),
        "size_bytes": str(path.stat().st_size),
        "signals": str(n_signals),
        "header_bytes": str(header_bytes),
        "records": str(n_records),
        "record_duration": record_duration,
        "labels": ", ".join(labels[:8]) + ("…" if len(labels) > 8 else ""),
    }


def _select_from_menu(title: str, options: list[str]) -> int | None:
    if not options:
        return None
    idx = _arrow_menu_select(title, options, default_index=0)
    if idx is not None:
        return idx
    return None


def browse_results(*, outputs: list[Path], title: str = "nicolet-e2edf") -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.syntax import Syntax
    from rich.table import Table

    console = Console()
    outputs = sorted({p for p in outputs if p.exists()}, key=lambda p: str(p))
    if not outputs:
        return

    def pause(message: str = "Press Enter to continue") -> None:
        try:
            console.input(f"[dim]{message}[/dim]")
        except (EOFError, KeyboardInterrupt):
            return

    def show_outputs() -> None:
        table = Table(title="Converted files", header_style="bold blue")
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("EDF", style="yellow")
        table.add_column("JSON", style="cyan")
        for idx, edf_path in enumerate(outputs, start=1):
            sidecar = edf_path.with_suffix(".json")
            table.add_row(str(idx), str(edf_path), str(sidecar) if sidecar.exists() else "—")
        console.print(Panel(table, title=title, border_style="blue"))

    def select_file() -> Path | None:
        names = [str(p) for p in outputs]
        chosen = _select_from_menu("Select file", names)
        if chosen is not None:
            return outputs[chosen]
        raw = Prompt.ask("Pick file number", default="1")
        try:
            idx = int(raw)
        except ValueError:
            return None
        if 1 <= idx <= len(outputs):
            return outputs[idx - 1]
        return None

    def view_json(edf_path: Path) -> None:
        sidecar = edf_path.with_suffix(".json")
        if not sidecar.exists():
            console.print(Panel(f"No sidecar found: {sidecar}", border_style="yellow"))
            pause()
            return
        text = sidecar.read_text(encoding="utf-8", errors="replace")
        with console.pager(styles=True):
            console.print(Panel(Syntax(text, "json", word_wrap=True), title=str(sidecar), border_style="cyan"))
        pause()

    def view_edf_summary(edf_path: Path) -> None:
        info = _read_edf_summary(edf_path)
        table = Table(title="EDF summary", show_header=False, box=None)
        for key, value in info.items():
            table.add_row(key, value)
        console.print(Panel(table, border_style="green"))
        pause()

    def snapshot_with_inspect(edf_path: Path) -> None:
        root = Path(__file__).resolve()
        repo_root = root.parents[2] if len(root.parents) >= 3 else None
        inspect_script = (repo_root / "inspect_edf.py") if repo_root else None
        if not inspect_script or not inspect_script.exists():
            console.print(
                Panel(
                    "Snapshot helper not found (inspect_edf.py). If you're running from an installed package,\n"
                    "use `mne.io.read_raw_edf(...)` manually or run the repo script.",
                    border_style="yellow",
                )
            )
            pause()
            return
        out_png = edf_path.with_suffix(".png")
        notch_choice = Prompt.ask("Notch filter (Hz)", choices=["50", "60", "0"], default="50")
        notch = float(notch_choice)
        console.print(Panel(f"Creating snapshot: {out_png}", border_style="blue"))
        
        # Check if MNE is available, if not use uv run --with mne (isolated, non-persistent)
        if mne_available():
            # MNE is available, run normally
            cmd = [sys.executable, str(inspect_script), str(edf_path), "--snapshot", str(out_png), "--notch", str(notch)]
        else:
            # MNE not available, use uv run --with mne (creates new isolated environment, non-persistent)
            # This works even when already running under uv run --isolated
            cmd = _build_uv_command(inspect_script, str(edf_path), "--snapshot", str(out_png), "--notch", str(notch))
        
        try:
            subprocess.run(  # noqa: S603
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            console.print(Panel(f"Snapshot failed:\n{exc.stdout}", border_style="red"))
            pause()
            return
        except FileNotFoundError:
            console.print(
                Panel(
                    "MNE not found and 'uv' command not available.\n"
                    "Install MNE with: uv run --with mne python inspect_edf.py ...",
                    border_style="red",
                )
            )
            pause()
            return
        console.print(Panel(f"Saved snapshot: {out_png}", border_style="green"))
        pause()

    def launch_mne_viewer(edf_path: Path) -> None:
        root = Path(__file__).resolve()
        repo_root = root.parents[2] if len(root.parents) >= 3 else None
        inspect_script = (repo_root / "inspect_edf.py") if repo_root else None
        if not inspect_script or not inspect_script.exists():
            console.print(Panel("inspect_edf.py not found; run from the repo to use the viewer.", border_style="yellow"))
            pause()
            return
        notch_choice = Prompt.ask("Notch filter (Hz)", choices=["50", "60", "0"], default="50")
        notch = float(notch_choice)
        console.print(Panel("Launching MNE viewer (close the window to return)…", border_style="blue"))
        
        # Check if MNE is available, if not use uv run --with mne (isolated, non-persistent)
        if mne_available():
            # MNE is available, run normally
            cmd = [sys.executable, str(inspect_script), str(edf_path), "--notch", str(notch)]
        else:
            # MNE not available, use uv run --with mne (creates new isolated environment, non-persistent)
            # This works even when already running under uv run --isolated
            cmd = _build_uv_command(inspect_script, str(edf_path), "--notch", str(notch))
        
        try:
            subprocess.run(  # noqa: S603
                cmd,
                check=False,
            )
        except FileNotFoundError:
            console.print(
                Panel(
                    "MNE not found and 'uv' command not available.\n"
                    "Install MNE with: uv run --with mne python inspect_edf.py ...",
                    border_style="red",
                )
            )
            pause()
            return
        pause("Press Enter to return to the browser")

    show_outputs()
    if not Confirm.ask("Browse results now?", default=True):
        return

    actions = [
        "List converted files",
        "View JSON sidecar",
        "Show EDF header summary",
        "Launch MNE viewer (inspect_edf.py)",
        "Create MNE snapshot PNG (inspect_edf.py)",
        "Quit",
    ]

    while True:
        choice = _select_from_menu("Choose an action", actions)
        if choice is None:
            raw = Prompt.ask(
                "Action",
                choices=[str(idx) for idx in range(1, len(actions) + 1)],
                default=str(len(actions)),
            )
            choice = int(raw) - 1

        if choice == 0:
            show_outputs()
            pause()
        elif choice == 1:
            selected = select_file()
            if selected:
                view_json(selected)
        elif choice == 2:
            selected = select_file()
            if selected:
                view_edf_summary(selected)
        elif choice == 3:
            selected = select_file()
            if selected:
                launch_mne_viewer(selected)
        elif choice == 4:
            selected = select_file()
            if selected:
                snapshot_with_inspect(selected)
        else:
            break


def run_tui(
    *,
    inputs: list[tuple[Path, Path | None, Path]],
    options: TuiOptions,
    convert_one,  # callable: (source_path, output_dir, output_path, input_root, status_cb) -> None
    title: str = "nicolet-e2edf",
) -> int:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.text import Text

    console = Console(force_terminal=True)

    total_files = len(inputs)
    file_progress = Progress(
        SpinnerColumn(style="magenta"),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    overall_task = file_progress.add_task("Overall", total=total_files)

    state: dict[str, str | None] = {"file": None, "stage": None, "error": None}

    def status_cb(stage: str) -> None:
        state["stage"] = stage

    def render() -> Panel:
        subtitle_parts: list[str] = []
        if options.resample_to is not None:
            subtitle_parts.append(f"resample→{options.resample_to:g}Hz")
        # Show filter info if any
        if options.lowcut is not None and options.highcut is not None:
            subtitle_parts.append(f"bandpass:{options.lowcut}–{options.highcut}Hz")
        elif options.lowcut is not None:
            subtitle_parts.append(f"highpass:{options.lowcut}Hz")
        elif options.highcut is not None:
            subtitle_parts.append(f"lowpass:{options.highcut}Hz")
        if options.notch is not None and options.notch > 0:
            subtitle_parts.append(f"notch:{options.notch}Hz")
        if options.json_sidecar:
            subtitle_parts.append("json-sidecar")
        if options.verbose:
            subtitle_parts.append("verbose")
        subtitle = " · ".join(subtitle_parts) if subtitle_parts else "raw (no filtering)"

        current = state["file"] or "—"
        stage = state["stage"] or "—"
        header = Text.assemble(
            ("EEG CONVERTER", "bold green"),
            ("  ", ""),
            (title, "bold white"),
            ("\n", ""),
            (subtitle, "dim"),
        )
        details = Text.assemble(
            ("Input: ", "bold"),
            (current, "yellow"),
            ("\nStage: ", "bold"),
            (stage, "cyan"),
        )
        if state["error"]:
            details.append("\n")
            details.append(Text(f"Last error: {state['error']}", style="bold red"))

        content = Group(header, "", EEGWave(width=None, height=8), "", details, "", file_progress)
        return Panel(content, border_style="blue", title="TUI", subtitle=subtitle)

    failures = 0
    with Live(render(), console=console, refresh_per_second=20, screen=True):
        for source_path, input_root, output_path in inputs:
            state["file"] = str(source_path)
            state["stage"] = "starting"
            state["error"] = None

            try:
                convert_one(
                    source_path=source_path,
                    output_path=output_path,
                    input_root=input_root,
                    status_cb=status_cb,
                )
            except Exception as exc:  # pragma: no cover - exercised via real runs
                failures += 1
                state["error"] = str(exc)
            finally:
                file_progress.advance(overall_task, 1)

    if failures:
        console.print(
            Panel(
                f"{failures}/{total_files} conversions failed. Re-run with `--verbose` for details.",
                border_style="red",
                title="Done (with errors)",
            )
        )
        return 1

    console.print(Panel(f"Converted {total_files} file(s).", border_style="green", title="Done"))
    browse_results(outputs=[out for _, _, out in inputs], title=title)
    return 0
