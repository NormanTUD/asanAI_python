from pathlib import Path
import re
import argparse
from rich.console import Console
from rich.progress import SpinnerColumn, Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.text import Text
from beartype import beartype
from typing import Optional, Union
import sys
from pprint import pprint
import shutil
import subprocess
import tempfile

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

@beartype
def convert_to_keras_if_needed() -> bool:
    keras_h5_file = 'model.h5'
    tfjs_model_json = 'model.json'

    # Check if conversion is needed
    if os.path.exists(keras_h5_file):
        console.print(f"[green]✔ Conversion not needed:[/] '{keras_h5_file}' already exists.")
        return True

    if not os.path.exists(tfjs_model_json):
        console.print(f"[yellow]⚠ Conversion not possible:[/] '{tfjs_model_json}' not found.")
        return True

    console.print(f"[cyan]ℹ Conversion needed:[/] '{keras_h5_file}' does not exist, but '{tfjs_model_json}' found.")

    # Helper function to check if a command exists in PATH
    def is_command_available(cmd: str) -> bool:
        return shutil.which(cmd) is not None

    conversion_args = [
        '--input_format=tfjs_layers_model',
        '--output_format=keras',
        tfjs_model_json,
        keras_h5_file
    ]

    if is_command_available('tensorflowjs_converter'):
        with console.status("[bold green]Local tensorflowjs_converter found. Starting conversion..."):
            cmd = ['tensorflowjs_converter'] + conversion_args
            try:
                completed_process = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                console.print(f"[green]✔ Local conversion succeeded.[/]")
                console.print(Text(completed_process.stdout.strip(), style="dim"))
                return True
            except subprocess.CalledProcessError as e:
                console.print("[red]✘ Local conversion failed:[/]")
                console.print(Text(e.stderr.strip(), style="bold red"))
                console.print("[yellow]➜ Falling back to Docker-based conversion...[/]")
    else:
        console.print("[yellow]⚠ tensorflowjs_converter CLI not found locally.[/]")

    if not is_command_available('docker'):
        console.print("[red]✘ Docker is not installed or not found in PATH. Cannot perform fallback conversion.[/]")
        return False

    try:
        subprocess.run(['docker', 'info'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        console.print("[red]✘ Docker daemon not running or inaccessible. Cannot perform fallback conversion.[/]")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = os.path.join(tmpdir, 'Dockerfile')

        dockerfile_content = '''FROM python:3.10-slim

RUN apt-get update && \\
    apt-get install -y --no-install-recommends build-essential curl && \\
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

RUN python -m pip install \\
    tensorflow==2.12.0 \\
    tensorflowjs==4.7.0 \\
    jax==0.4.13 \\
    jaxlib==0.4.13

WORKDIR /app

CMD ["/bin/bash"]
'''
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        image_name = 'tfjs_converter_py310_dynamic'

        console.print("[cyan]ℹ Building Docker image for fallback conversion...[/]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=True,
            console=console
        ) as progress:
            build_task = progress.add_task("Building Docker image...", total=None)
            try:
                build_cmd = ['docker', 'build', '-t', image_name, '-f', dockerfile_path, tmpdir]
                subprocess.run(build_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                progress.update(build_task, description="Docker image built successfully.")
            except subprocess.CalledProcessError as e:
                progress.stop()
                console.print("[red]✘ Docker build failed with error:[/]")
                console.print(Text(e.stderr.strip(), style="bold red"))
                return False

        run_cmd = [
            'docker', 'run', '--rm',
            '-v', f"{os.path.abspath(os.getcwd())}:/app",
            image_name,
            'tensorflowjs_converter',
        ] + conversion_args

        with console.status("[bold green]Running conversion inside Docker container..."):
            try:
                run_process = subprocess.run(run_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                console.print("[green]✔ Conversion inside Docker container succeeded.[/]")
                console.print(Text(run_process.stdout.strip(), style="dim"))
            except subprocess.CalledProcessError as e:
                console.print("[red]✘ Conversion inside Docker container failed with error:[/]")
                console.print(Text(e.stderr.strip(), style="bold red"))
                return False

    return True
