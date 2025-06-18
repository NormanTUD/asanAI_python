import sys
from pprint import pprint
import re
import os
from pathlib import Path
import tempfile
import subprocess
from typing import Optional, Union, Any
import shutil

import numpy as np
import cv2
from skimage import transform
from PIL import Image
from rich.console import Console
from rich.progress import SpinnerColumn, Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.text import Text
from beartype import beartype

def dier (msg: Any) -> None:
    pprint(msg)
    sys.exit(1)

console = Console()

@beartype
def _newest_match(directory: Union[Path, str], pattern: str) -> Optional[Path]:
    directory = Path(directory)

    candidates = [
        p for p in directory.iterdir()
        if re.fullmatch(pattern, p.name)
    ]

    if not candidates:
        return None

    def extract_number(p: Path) -> int:
        match = re.search(r"\((\d+)\)", p.name)
        if not match:
            raise ValueError(f"No number found in parentheses in: {p.name}")
        return int(match.group(1))

    candidates.sort(
        key=extract_number,
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
        return False

    if not os.path.exists(tfjs_model_json):
        console.print(f"[yellow]⚠ Conversion not possible:[/] '{tfjs_model_json}' not found.")
        return False

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
                console.print("[green]✔ Local conversion succeeded.[/]")
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
        with open(dockerfile_path, mode='w', encoding="utf-8") as f:
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

@beartype
def load(filename: Union[Path, str], width: int = 224, height: int = 224, divideby: float = 255.0) -> np.ndarray:
    image = Image.open(filename)
    np_image: np.ndarray = np.array(image).astype('float32') / divideby
    np_image = transform.resize(np_image, (height, width, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

@beartype
def load_frame(frame: np.ndarray, width: int = 224, height: int = 224, divideby: float = 255.0) -> np.ndarray:
    np_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # pylint: disable=no-member
    np_image = np.array(np_image).astype('float32') / divideby
    np_image = transform.resize(np_image, (height, width, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

@beartype
def annotate_frame(frame: np.ndarray, predictions: np.ndarray, labels: list[str]) -> np.ndarray:
    probs = predictions[0]
    best_idx = int(np.argmax(probs))

    for i, label in enumerate(labels):
        text = f"{label}: {probs[i]:.2f}"
        colour = (0, 255, 0) if i == best_idx else (255, 0, 0)
        cv2.putText( # pylint: disable=no-member
            frame,
            text,
            (10, 30 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX, # pylint: disable=no-member
            0.8,
            colour,
            2,
        )
    return frame
