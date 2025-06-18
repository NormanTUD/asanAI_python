from pathlib import Path
import re
import argparse
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
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
        print(f"Conversion not needed: '{keras_h5_file}' already exists.")
        return True

    if not os.path.exists(tfjs_model_json):
        print(f"Conversion not possible: '{tfjs_model_json}' not found.")
        return True

    print(f"Conversion needed: '{keras_h5_file}' does not exist, but '{tfjs_model_json}' found.")

    # Helper function to check if a command exists in PATH
    def is_command_available(cmd):
        return shutil.which(cmd) is not None

    # Command base args for conversion to keras h5
    conversion_args = [
        '--input_format=tfjs_layers_model',
        '--output_format=keras',
        tfjs_model_json,
        keras_h5_file
    ]

    # Try local tensorflowjs_converter CLI first
    if is_command_available('tensorflowjs_converter'):
        print("tensorflowjs_converter CLI found locally, trying local conversion...")

        cmd = ['tensorflowjs_converter'] + conversion_args

        try:
            completed_process = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("Local conversion succeeded.")
            print(completed_process.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print("Local conversion failed with error:")
            print(e.stderr)
            print("Falling back to Docker-based conversion...")
    else:
        print("tensorflowjs_converter CLI not found locally.")

    # Check Docker availability
    if not is_command_available('docker'):
        print("Docker is not installed or not found in PATH. Cannot perform fallback conversion.")
        return False

    try:
        subprocess.run(['docker', 'info'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Docker daemon does not seem to be running or accessible. Cannot perform fallback conversion.")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = os.path.join(tmpdir, 'Dockerfile')

        dockerfile_content = '''FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

RUN python -m pip install \
    tensorflow==2.12.0 \
    tensorflowjs==4.7.0 \
    jax==0.4.13 \
    jaxlib==0.4.13

WORKDIR /app

CMD ["/bin/bash"]
'''
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        image_name = 'tfjs_converter_py310_dynamic'

        print("Building Docker image for fallback conversion...")
        try:
            build_cmd = ['docker', 'build', '-t', image_name, '-f', dockerfile_path, tmpdir]
            subprocess.run(build_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Docker image build succeeded.")
        except subprocess.CalledProcessError as e:
            print("Docker build failed with error:")
            print(e.stderr)
            return False

        # Run conversion in docker with keras output (.h5)
        run_cmd = [
            'docker', 'run', '--rm',
            '-v', f"{os.path.abspath(os.getcwd())}:/app",
            image_name,
            'tensorflowjs_converter',
        ] + conversion_args

        print("Running conversion inside Docker container. This can take a while...")

        try:
            run_process = subprocess.run(run_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Conversion inside Docker container succeeded.")
            print(run_process.stdout)
        except subprocess.CalledProcessError as e:
            print("Conversion inside Docker container failed with error:")
            print(e.stderr)
            return False

    return True
