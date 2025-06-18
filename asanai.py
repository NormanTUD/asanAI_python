from pathlib import Path
import re
import argparse
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from beartype import beartype
from typing import Optional, Union
import sys
from pprint import pprint

def dier (msg):
    pprint(msg)
    sys.exit(1)

console = Console()

@beartype
def _newest_match(directory: Union[Path, str], pattern: str) -> Path | None:
    """Return the newest Path in *directory* matching *pattern* or None."""
    candidates = [
        p for p in directory.iterdir()
        if re.fullmatch(pattern, p.name)
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: int(re.search(r"\((\d+)\)", p.name).group(1)),
        reverse=True,
    )
    return candidates[0]

@beartype
def rename_model_files_if_needed(directory: Optional[Union[Path, str]]) -> None:
    if directory is None:
        console.log("[red]No directory provided[/red]")
        return

    directory = Path(directory)

    jobs: tuple[tuple[str, str], ...] = (
        ("model.json",        r"model\((\d+)\)\.json"),
        ("model.weights.bin", r"model\.weights\((\d+)\)\.bin"),
    )

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task_ids = {
            canonical: progress.add_task(f"Checking {canonical}", total=1)
            for canonical, _ in jobs
        }

        for canonical, regex in jobs:
            progress.update(task_ids[canonical], advance=0)  # render row
            target = directory / canonical
            if target.exists():
                progress.update(task_ids[canonical], completed=1)
                console.log(f"[green]{canonical} already present[/green]")
                continue

            newest = _newest_match(directory, regex)
            if newest:
                newest.rename(target)
                console.log(f"[green]Renamed[/green] {newest.name} → {canonical}")
            else:
                console.log(f"[yellow]Warning:[/yellow] No candidate for {canonical}")
            progress.update(task_ids[canonical], completed=1)
