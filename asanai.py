# pylint: disable=too-many-lines

import sys

try:
    import pprint
    import re
    import os
    from pathlib import Path
    import tempfile
    import subprocess
    from typing import Optional, Union, Any, Tuple
    import shutil
    from importlib import import_module, util
    import json
    from types import ModuleType
    import platform
    import traceback
    import urllib.request
    import urllib.error
    import psutil
    import unicodedata

    from colorama import Style, Fore, Back, init
    import numpy as np
    import cv2
    from skimage import transform
    from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.progress import SpinnerColumn, Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.text import Text
    from beartype import beartype
except ModuleNotFoundError as e:
    print(f"Failed ot load module: {e}")
    sys.exit(1)

init(autoreset=True)

@beartype
def dier (msg: Any) -> None:
    pprint.pprint(msg)
    sys.exit(1)

console = Console()

@beartype
def print_predictions_line(predictions: np.ndarray, labels: list) -> None:
    vals = predictions[0]
    max_index = int(np.argmax(vals))  # Index des höchsten Werts

    parts = []
    for i, (label, value) in enumerate(zip(labels, vals)):
        if i == max_index:
            part = f"{Style.BRIGHT}{Fore.WHITE}{Back.GREEN}{label}: {value:.10f}{Style.RESET_ALL}"
        else:
            part = f"{label}: {value:.10f}"
        parts.append(part)

    line = "  ".join(parts)
    sys.stdout.write("\r" + line + " " * 5)
    sys.stdout.flush()

@beartype
def _pip_install(package: str, quiet: bool = False) -> bool:
    if not _pip_available():
        console.print("[red]pip is not available – cannot install packages automatically.[/red]")
        return False

    cmd = [sys.executable, "-m", "pip", "install", package]
    if quiet:
        cmd.append("-q")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Installing {package}...[/cyan]"),
            transient=True,
            console=console,
        ) as progress:
            task = progress.add_task("pip_install", start=False)
            progress.start_task(task)
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                console.print(f"[red]Failed to install {package}.[/red]")
                console.print(f"[red]{result.stderr.strip()}[/red]")
            return result.returncode == 0
    except FileNotFoundError:
        console.print(f"[red]Python executable not found: {sys.executable}[/red]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Installation failed for {package} (non-zero exit).[/red]")
        console.print(f"[red]{e.stderr.strip()}[/red]")
    except subprocess.SubprocessError as e:
        console.print(f"[red]A subprocess error occurred during installation of {package}.[/red]")
        console.print(f"[red]{str(e).strip()}[/red]")
    except KeyboardInterrupt:
        console.print(f"[yellow]Installation of {package} interrupted by user.[/yellow]")

    return False

@beartype
def rule(msg) -> None:
    console.rule(f"{msg}")

@beartype
def _in_virtual_env() -> bool:
    return (
        # virtualenv / venv
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        or hasattr(sys, "real_prefix")
        # conda
        or bool(os.environ.get("CONDA_PREFIX"))
    )

@beartype
def _pip_available() -> bool:
    return shutil.which("pip") is not None or util.find_spec("pip") is not None

@beartype
def _proxy_hint() -> None:
    if not (os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")):
        console.print(
            "[yellow]No HTTP(S)_PROXY found – if you’re behind a proxy or corporate "
            "firewall, set HTTP_PROXY / HTTPS_PROXY or pass --proxy to pip.[/yellow]"
        )

@beartype
def _gpu_hint() -> None:
    if shutil.which("nvidia-smi"):
        console.print("[green]CUDA‑capable GPU detected via nvidia‑smi.[/green]")
    elif platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
        console.print(
            "[yellow]Apple Silicon detected. "
            "For GPU acceleration install [bold]tensorflow-metal[/bold] as well.[/yellow]"
        )
    else:
        console.print(
            "[yellow]No GPU detected (or drivers missing). "
            "CPU builds will run, but it will be slower than with GPU.[/yellow]"
        )

@beartype
def _platform_wheel_warning() -> None:
    sys_name = platform.system()
    arch = platform.machine().lower()

    if sys_name == "Darwin" and arch in {"arm64", "aarch64"}:
        console.print(
            "[yellow]ARM macOS: Regular 'tensorflow' wheels don’t work – "
            "falling back to [bold]tensorflow-macos[/bold].[/yellow]"
        )
    elif sys_name == "Linux" and arch not in {"x86_64", "amd64"}:
        console.print(
            "[red]Warning: Pre‑built TensorFlow wheels for this CPU architecture "
            "may not exist. Manual build might be required.[/red]"
        )
    elif sys_name == "Windows" and arch not in {"amd64", "x86_64"}:
        console.print(
            "[red]Warning: Non‑64‑bit Windows or uncommon architectures are "
            "not supported by official TensorFlow wheels.[/red]"
        )

@beartype
def download_file(url: str, dest_path: str) -> bool:
    try:
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                print(f"Error: Server responded with status {response.status}")
                return False
            data = response.read()
            with open(dest_path, 'wb') as file:
                file.write(data)
        print(f"File successfully downloaded: {dest_path}")
        return True
    except urllib.error.HTTPError as e:
        print(f"HTTP error while downloading: {e.code} - {e.reason}")
        return False

    except urllib.error.URLError as e:
        print(f"URL error while downloading: {e.reason}")
        return False

    except ValueError as e:
        print(f"Invalid URL: {e}")
        return False

@beartype
def run_installer(installer_path: str) -> bool:
    try:
        # subprocess.run waits for the process to finish
        result = subprocess.run([installer_path, '/install', '/quiet', '/norestart'], check=False)
        if result.returncode == 0:
            print("Installation completed successfully.")
            return True

        print(f"Installation failed with error code {result.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"Installer file not found: {e}")
        return False

    except PermissionError as e:
        print(f"Permission denied when running installer: {e}")
        return False

    except subprocess.SubprocessError as e:
        print(f"Subprocess error while running installer: {e}")
        return False

@beartype
def normalize_input(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()

@beartype
def ask_yes_no(prompt) -> bool:
    if os.environ.get("CI") is not None:
        return True

    while True:
        answer = normalize_input(Prompt.ask(prompt, default="no")).strip().lower()

        console.print(f"Answer: {answer}")

        if answer in ['yes', 'y', 'j']:
            return True

        if answer in ['no', 'n', 'nein']:
            return False

        console.print("[red]Please answer with 'yes', 'y' or 'no', 'n'.[/red]")

@beartype
def download_and_install_ms_visual_cpp() -> None:
    url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    filename = "vc_redist.x64.exe"
    filepath = os.path.join(os.getcwd(), filename)

    print("This Visual C++ Redistributable package is required for TensorFlow to work properly.")
    continue_install = ask_yes_no("Do you want to download and install it now? (yes/j/y/no): ")
    if not continue_install:
        print("Operation cancelled by user.")
        sys.exit(0)

    print("Starting file download...")
    success = download_file(url, filepath)
    if not success:
        print("Download failed, aborting.")
        sys.exit(1)

    print("Starting installation...")
    success = run_installer(filepath)
    if not success:
        print("Installation failed.")
        sys.exit(1)

@beartype
def install_tensorflow(full_argv: Optional[list] = None) -> Optional[ModuleType]:
    console.rule("[bold cyan]Checking for TensorFlow…[/bold cyan]")

    with console.status("Fast-probing TensorFlow Module. Will load and return it if it exists."):
        if util.find_spec("tensorflow"):
            tf = import_module("tensorflow")  # full import only when needed
            _gpu_hint()
            return tf

    console.print("[yellow]TensorFlow not found. Installation required.[/yellow]")

    # Safety: insist on an env
    if not _in_virtual_env():
        console.print(
            "[red]You must activate a virtual environment (venv or conda) "
            "before installing TensorFlow.[/red]"
        )
        sys.exit(1)

    _platform_wheel_warning()

    # Choose package name based on platform
    pkg_name = "tensorflow"
    if platform.system() == "Windows":
        download_and_install_ms_visual_cpp()

    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        pkg_name = "tensorflow-macos"

    if _pip_install(pkg_name):
        _gpu_hint()
    elif _pip_install("tf-nightly"):
        console.print("[yellow]Falling back to nightly build.[/yellow]")
        _gpu_hint()
    else:
        venv_path = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX") or sys.prefix
        activate_hint = ""

        if platform.system() == "Windows":
            bat_path = os.path.join(venv_path, "Scripts", "activate.bat")
            ps1_path = os.path.join(venv_path, "Scripts", "Activate.ps1")
            activate_hint = (
                f"\n[bold]CMD:[/bold]      {bat_path}\n"
                f"[bold]PowerShell:[/bold] {ps1_path}"
            )
        else:
            sh_path = os.path.join(venv_path, "bin", "activate")
            activate_hint = f"\n[bold]Bash/zsh:[/bold] source {sh_path}"

        console.print(
            "[red]Automatic installation failed.[/red]\n"
            "[yellow]Please install TensorFlow manually inside your virtual environment.[/yellow]"
            f"{activate_hint}"
        )

        sys.exit(1)

    console.print("[green]TensorFlow installed successfully! Trying to restart the script automatically...[/green]")

    if full_argv is not None and isinstance(full_argv, list):
        os.execv(sys.executable, [sys.executable] + full_argv)
    else:
        console.print("You need to manually restart your script after TensorFlow was installed.")
        sys.exit(0)

    return None

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
def find_model_files(directory: Optional[Union[Path, str]] = ".") -> dict[str, Optional[Path]]:
    if directory is None:
        console.log("[red]No directory provided[/red]")
        return {}

    directory = Path(directory)

    jobs: tuple[tuple[str, str], ...] = (
        ("model.json",        r"model\((\d+)\)\.json"),
        ("model.weights.bin", r"model\.weights\((\d+)\)\.bin"),
    )

    found_files: dict[str, Optional[Path]] = {}

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
                console.log(f"[green]{canonical} found[/green]")
                found_files[canonical] = target
                continue

            newest = _newest_match(directory, regex)
            if newest:
                console.log(f"[yellow]Using[/yellow] {newest.name} instead of {canonical}")
                found_files[canonical] = newest
            else:
                console.log(f"[red]Missing:[/red] No match for {canonical}")
                found_files[canonical] = None
            progress.update(task_ids[canonical], completed=1)

    return found_files

@beartype
def _is_command_available(cmd: str) -> bool:
    return shutil.which(cmd) is not None

@beartype
def _pip_install_tensorflowjs_converter_and_run_it(conversion_args: list) -> bool:
    if  not _is_command_available('tensorflowjs_converter'):
        _pip_install("tensorflowjs", True)

    if  _is_command_available('tensorflowjs_converter'):
        if _is_command_available('tensorflowjs_converter'):
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
                except KeyboardInterrupt:
                    console.print("[green]You cancelled the conversion progress by CTRL-C. You need to run this script again or do it manually for this program to work.[/green]")
                    sys.exit(0)
        else:
            console.print("[yellow]⚠ tensorflowjs_converter CLI not found locally.[/]")
    else:
        if platform.system() == "Windows":
            console.print("[yellow]⚠ Installing tensorflowjs module failed. Trying to fall back to docker. This can take some time, but only has to be done once. Start docker-desktop once before restarting the new cmd.[/]")
        else:
            console.print("[yellow]⚠ Installing tensorflowjs module failed. Trying to fall back to docker. This can take some time, but only has to be done once.[/]")

    return False

@beartype
def copy_and_patch_tfjs(model_json_path: str, weights_bin_path: str, out_prefix: str = "tmp_model") -> Tuple[str, str]:
    json_out = f"{out_prefix}.json"
    bin_out  = f"{out_prefix}.bin"

    # --- patch JSON ---
    with open(model_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Point every manifest entry to the newly created .bin
    for manifest in data.get("weightsManifest", []):
        manifest["paths"] = [f"./{Path(bin_out).name}"]

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # --- copy BIN ---
    shutil.copyfile(weights_bin_path, bin_out)

    return json_out, bin_out

@beartype
def delete_tmp_files(json_file, bin_file) -> None:
    if os.path.exists(json_file):
        with console.status(f"[bold green]Deleting {json_file}..."):
            os.unlink(json_file)

    if os.path.exists(bin_file):
        with console.status(f"[bold green]Deleting {bin_file}..."):
            os.unlink(bin_file)

@beartype
def is_docker_installed():
    return shutil.which("docker") is not None

@beartype
def try_install_docker_linux():
    if shutil.which('apt'):
        console.print("[yellow]🛠 Installing Docker with apt...[/yellow]")
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        subprocess.run(['sudo', 'apt', 'install', '-y', 'docker.io'], check=True)
    elif shutil.which('dnf'):
        console.print("[yellow]🛠 Installing Docker with dnf...[/yellow]")
        subprocess.run(['sudo', 'dnf', 'install', '-y', 'docker'], check=True)
    elif shutil.which('pacman'):
        console.print("[yellow]🛠 Installing Docker with pacman...[/yellow]")
        subprocess.run(['sudo', 'pacman', '-Sy', '--noconfirm', 'docker'], check=True)
    else:
        console.print("[red]❌ Unsupported Linux package manager.[/red]")
        console.print("👉 Install manually: https://docs.docker.com/engine/install/")

@beartype
def try_install_docker_windows():
    if not shutil.which('winget'):
        print("❌ Winget not found. Install Docker manually:")
        print("👉 https://docs.docker.com/docker-for-windows/install/")
        return

    print("🛠 Installing Docker Desktop using winget...")
    try:
        subprocess.run([
            'winget', 'install', '--id', 'Docker.DockerDesktop',
            '--source', 'winget',
            '--accept-package-agreements',
            '--accept-source-agreements'
        ], check=True)
        print("✅ Docker installation started. Please complete setup manually if needed.")
        print("✅ Please restart the cmd on Windows and restart script manually.")
        start_docker_if_not_running()
        sys.exit(0)

    except subprocess.CalledProcessError as e:
        print("❌ Docker installation failed. Manual install:")
        print("👉 https://docs.docker.com/docker-for-windows/install/")
        print(f"Details: {e}")

@beartype
def try_install_docker_mac():
    if shutil.which("brew"):
        print("🛠 Installing Docker via Homebrew...")
        subprocess.run(['brew', 'install', '--cask', 'docker'], check=True)
        print("✅ Docker installed. Please start Docker Desktop manually.")
    else:
        print("❌ Homebrew not found.")
        print("👉 Install manually: https://docs.docker.com/docker-for-mac/install/")

@beartype
def update_wsl_if_windows() -> None: # pylint: disable=too-many-branches
    if platform.system() != "Windows":
        return

    console.print("[bold green]Windows system detected.[/bold green] Checking for WSL...")

    try:
        subprocess.run(["wsl", "--status"], capture_output=True, text=True, check=True)
        console.print("[green]WSL is installed.[/green]")
    except FileNotFoundError:
        console.print("[yellow]WSL is not installed or not in PATH. Attempting to install...[/yellow]")
        try:
            subprocess.run(["wsl", "--install"], capture_output=True, text=True, check=True)
            console.print("[bold green]✅ WSL installation initiated successfully.[/bold green]")
            console.print("[cyan]You may need to reboot your system to complete the installation.[/cyan]")
        except FileNotFoundError as e:
            console.print("[red]❌ 'wsl' command not found. Is WSL supported on your system?[/red]")
            console.print(f"[red]{str(e)}[/red]")

        except PermissionError as e:
            console.print("[red]❌ Permission denied. Try running this script with administrative privileges.[/red]")
            console.print(f"[red]{str(e)}[/red]")

        except subprocess.CalledProcessError as e:
            console.print("[red]❌ WSL installation failed with a subprocess error:[/red]")
            console.print(f"[red]Exit code: {e.returncode}[/red]")
            console.print(f"[red]Command: {' '.join(e.cmd)}[/red]")
            if e.stderr:
                console.print(f"[red]Error Output: {e.stderr.strip()}[/red]")
            else:
                console.print("[red]No error output available.[/red]")

        except OSError as e:
            console.print("[red]❌ Operating system error during WSL installation.[/red]")
            console.print(f"[red]{str(e)}[/red]")

        return
    except subprocess.CalledProcessError as e:
        console.print("[red]❌ Error while checking WSL status:[/red]")
        console.print(f"[red]{e.stderr.strip()}[/red]")
        return

    # Check if update is available
    console.print("[bold cyan]Checking if a WSL update is available...[/bold cyan]")
    try:
        check_update = subprocess.run(["wsl", "--update", "--status"], capture_output=True, text=True, check=True)
        if "The installed version is the same as the latest version" in check_update.stdout:
            console.print("[green]✅ WSL is already up to date.[/green]")
            return
        console.print("[yellow]⚠ An update for WSL is available.[/yellow]")

    except FileNotFoundError as e:
        console.print("[red]❌ 'wsl' command not found. Ensure WSL is installed and available in PATH.[/red]")
        console.print(f"[red]{str(e)}[/red]")
        return

    except PermissionError as e:
        console.print("[red]❌ Permission denied. You may need to run this script as administrator.[/red]")
        console.print(f"[red]{str(e)}[/red]")
        return

    except subprocess.CalledProcessError as e:
        console.print("[red]❌ Error checking WSL update status:[/red]")
        console.print(f"[red]Exit code: {e.returncode}[/red]")
        console.print(f"[red]Command: {' '.join(e.cmd)}[/red]")
        if e.stderr:
            console.print(f"[red]Error Output: {e.stderr.strip()}[/red]")
        else:
            console.print("[red]No error output available.[/red]")
        return

    except OSError as e:
        console.print("[red]❌ Operating system error occurred while checking for WSL updates.[/red]")
        console.print(f"[red]{str(e)}[/red]")
        return

    if ask_yes_no("Do you want to run 'wsl --update' now? [y/j/yes]: "):
        try:
            console.print("\n[bold cyan]Running 'wsl --update'...[/bold cyan]")
            update = subprocess.run(["wsl", "--update"], capture_output=True, text=True, check=True)
            if update.returncode == 0:
                console.print("[bold green]✅ WSL was successfully updated.[/bold green]")
            else:
                console.print("[bold red]❌ Error during 'wsl --update':[/bold red]")
                console.print(f"[red]{update.stderr.strip()}[/red]")
        except subprocess.CalledProcessError as e:
            console.print("[bold red]❌ Error during 'wsl --update' command execution:[/bold red]")
            console.print(f"[red]{e.stderr.strip() if e.stderr else str(e)}[/red]")
        except FileNotFoundError as e:
            console.print("[bold red]❌ 'wsl' executable not found. Is WSL installed?[/bold red]")
            console.print(f"[red]{str(e)}[/red]")
        except subprocess.TimeoutExpired as e:
            console.print("[bold red]❌ 'wsl --update' command timed out.[/bold red]")
            console.print(f"[red]{str(e)}[/red]")
        except OSError as e:
            console.print("[bold red]❌ OS error occurred while running 'wsl --update':[/bold red]")
            console.print(f"[red]{str(e)}[/red]")
    else:
        console.print("[yellow]Update cancelled. WSL remains unchanged.[/yellow]")

@beartype
def try_install_docker():
    if is_docker_installed():
        print("✅ Docker is already installed.")
        return True

    answer = ask_yes_no("Do you want to try installing Docker? [y/j/yes]: ")

    if not answer:
        console.print("[red]Docker is required. The script cannot continue without Docker.[/red]")

        return False

    console.print("[green]Proceeding with Docker installation. This may ask you for your user password...[/green]")

    system = platform.system()
    console.print(f"[yellow]🔍 Detected OS: {system}[/yellow]")

    if system == 'Linux':
        try_install_docker_linux()
    elif system == 'Windows':
        try_install_docker_windows()
    elif system == 'Darwin':
        try_install_docker_mac()
    else:
        print(f"❌ Unsupported OS: {system}")
        print("👉 Install manually: https://docs.docker.com/get-docker/")
        return False

    if is_docker_installed():
        print("✅ Docker installation successful.")
        return True

    print("⚠ Docker still not found. Please install manually:")
    print("👉 https://docs.docker.com/get-docker/")
    return False

@beartype
def check_docker_and_try_to_install(tfjs_model_json: str, weights_bin: str) -> bool:
    if not _is_command_available('docker'):
        if not try_install_docker():
            delete_tmp_files(tfjs_model_json, weights_bin)
            return False

        if not _is_command_available('docker'):
            console.print("[red]✘ Installing Docker automatically failed.[/]")
            delete_tmp_files(tfjs_model_json, weights_bin)
            return False

    return True

@beartype
def is_windows() -> bool:
    return platform.system().lower() == "windows"

@beartype
def get_program_files() -> str:
    program_w6432 = os.environ.get("ProgramW6432")
    if program_w6432 is not None:
        return program_w6432

    program_files = os.environ.get("ProgramFiles")
    if program_files is not None:
        return program_files

    raise EnvironmentError("Neither 'ProgramW6432' nor 'ProgramFiles' environment variables are set.")

@beartype
def is_docker_running() -> bool:
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and "Docker Desktop.exe" in proc.info['name']:
            return True
    return False

@beartype
def start_docker() -> int:
    """
    Attempts to start Docker Desktop.exe from the Program Files directory.
    Returns:
        0 - Success
        2 - Program Files path not found
        3 - Docker Desktop.exe not found
        4 - Other errors (permission, OS, file not found, value error)
    """
    pf = get_program_files()
    if not pf:
        console.print("[bold red]❌ Program Files directory not found.[/bold red]")
        return 2

    path = os.path.join(pf, "Docker", "Docker", "Docker Desktop.exe")
    if not os.path.isfile(path):
        console.print(f"[bold red]❌ Docker Desktop executable not found at:[/bold red] {path}")
        return 3

    try:
        # Use 'with' to ensure resource cleanup, though Docker Desktop runs asynchronously.
        with subprocess.Popen([path], shell=False):
            # Not waiting for process completion to avoid blocking,
            # just start the process and immediately return success.
            return 0
    except FileNotFoundError as fnf_error:
        console.print(f"[bold red]❌ File or executable not found:[/bold red] {path}")
        console.print(f"[red]{str(fnf_error)}[/red]")
        return 4
    except PermissionError as perm_error:
        console.print(f"[bold red]❌ Permission denied to execute:[/bold red] {path}")
        console.print(f"[red]{str(perm_error)}[/red]")
        return 4
    except OSError as os_error:
        console.print(f"[bold red]❌ OS error occurred while trying to launch:[/bold red] {path}")
        console.print(f"[red]{str(os_error)}[/red]")
        return 4
    except ValueError as val_error:
        console.print(f"[bold red]❌ Invalid argument passed to Popen for:[/bold red] {path}")
        console.print(f"[red]{str(val_error)}[/red]")
        return 4

@beartype
def start_docker_if_not_running() -> bool:
    if not is_windows():
        return True
    if is_docker_running():
        return False
    return start_docker() == 0

@beartype
def convert_to_keras_if_needed(directory: Optional[Union[Path, str]] = ".") -> bool:
    keras_h5_file = 'model.h5'

    if os.path.exists(keras_h5_file):
        console.print(f"[green]✔ Conversion not needed:[/] '{keras_h5_file}' already exists.")
        return True

    rule("[bold cyan]Trying to convert downloaded model files[/]")

    files = find_model_files(directory)

    original_tfjs_model_json = str(files.get("model.json"))
    original_weights_bin = str(files.get("model.weights.bin"))

    if not os.path.exists(original_tfjs_model_json) or not os.path.exists(original_weights_bin):
        console.print("[red]No model.json and/or model.weights.bin found. Cannot continue. Have you downloaded the models from asanAI? If not, do so and put them in the same folder as your script.[/red]")
        sys.exit(1)

    if not os.path.exists(original_tfjs_model_json):
        console.print(f"[yellow]⚠ Conversion not possible:[/] '{original_tfjs_model_json}' not found.")
        return False

    tfjs_model_json, weights_bin = copy_and_patch_tfjs(original_tfjs_model_json, original_weights_bin)

    if not tfjs_model_json or not weights_bin:
        console.log("[red]Missing model files. Conversion aborted.[/red]")
        delete_tmp_files(tfjs_model_json, weights_bin)
        return False

    console.print(f"[cyan]Conversion needed:[/] '{keras_h5_file}' does not exist, but '{original_tfjs_model_json}' found.")

    conversion_args = [
        '--input_format=tfjs_layers_model',
        '--output_format=keras',
        tfjs_model_json,
        keras_h5_file
    ]

    if _pip_install_tensorflowjs_converter_and_run_it(conversion_args):
        delete_tmp_files(tfjs_model_json, weights_bin)
        return True

    update_wsl_if_windows()

    if not check_docker_and_try_to_install(tfjs_model_json, weights_bin):
        return False

    start_docker_if_not_running()

    try:
        subprocess.run(['docker', 'info'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        console.print("[red]✘ Docker daemon not running or inaccessible. Cannot perform fallback conversion.[/]")
        delete_tmp_files(tfjs_model_json, weights_bin)
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

        console.print("[cyan]Building Docker image for fallback conversion. This may take some time...[/]")
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
                delete_tmp_files(tfjs_model_json, weights_bin)
                return False
            except KeyboardInterrupt:
                progress.stop()
                console.print("[red]✘ Docker build was cancelled by CTRL-C[/]")
                delete_tmp_files(tfjs_model_json, weights_bin)
                sys.exit(0)

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
                delete_tmp_files(tfjs_model_json, weights_bin)
                return False
            except KeyboardInterrupt:
                console.print("[red]✘ Docker build was cancelled by CTRL-C[/]")
                delete_tmp_files(tfjs_model_json, weights_bin)
                sys.exit(0)

    delete_tmp_files(tfjs_model_json, weights_bin)

    return True

@beartype
def load(filename: Union[Path, str], height: int = 224, width: int = 224, divide_by: Union[int, float] = 255.0) -> Optional[np.ndarray]:
    rule(f"[bold cyan]Loading image {filename}[/]")
    try:
        if not os.path.exists(filename):
            console.print(f"[red]Error: The path '{filename}' could not be found![/red]")
            return None

        try:
            with console.status(f"Loading image {filename}"):
                image = Image.open(filename)

            with console.status(f"Converting image {filename} to numpy array and normalizing"):
                np_image: np.ndarray = np.array(image).astype('float32') / divide_by

            with console.status(f"Resizing image {filename} to (height = {height}, width = {width}, channels = 3)"):
                np_image = transform.resize(np_image, (height, width, 3))

            with console.status(f"Expanding numpy array dimensions from image {filename}"):
                np_image = np.expand_dims(np_image, axis=0)

            return np_image

        except PermissionError:
            console.print(f"[red]Error: Permission denied for file '{filename}'. Please check file permissions.[/red]")

        except UnidentifiedImageError:
            console.print(f"[red]Error: The file '{filename}' is not a valid image or is corrupted.[/red]")

        except ValueError as ve:
            console.print(f"[red]Error: ValueError encountered: {ve}. Possibly wrong image dimensions or resize parameters.[/red]")

        except TypeError as te:
            console.print(f"[red]Error: TypeError encountered: {te}. Check if 'divide_by' is a number (int or float).[/red]")

        except OSError as ose:
            console.print(f"[red]Error: OS error occurred: {ose}. Possible file system issue.[/red]")
    except KeyboardInterrupt:
        console.print(f"[green]You cancelled loading the image {filename} by pressing CTRL-C[/green]")
        sys.exit(0)

    return None

@beartype
def load_frame(frame: np.ndarray, height: int = 224, width: int = 224, divide_by: Union[int, float] = 255.0) -> Optional[np.ndarray]:
    try:
        np_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
        np_image = np.array(np_image).astype('float32') / divide_by
        np_image = transform.resize(np_image, (height, width, 3))
        np_image = np.expand_dims(np_image, axis=0)
        return np_image

    except cv2.error as e: # pylint: disable=no-member
        console.print(f"[red]OpenCV error during color conversion: {e}[/red]")

    except ValueError as ve:
        console.print(f"[red]ValueError during resize or processing: {ve}[/red]")

    except TypeError as te:
        console.print(f"[red]TypeError encountered: {te}. Check input types.[/red]")

    except OSError as ose:
        console.print(f"[red]OS error occurred: {ose}.[/red]")

    except KeyboardInterrupt:
        console.print("[green]You cancelled loading the fame by pressing CTRL-C[/green]")
        sys.exit(0)

    return None

@beartype
def _format_probabilities(values: np.ndarray) -> list[str]:
    for precision in range(3, 12):  # vernünftiger Bereich
        formatted = [f"{v:.{precision}f}" for v in values]
        if len(set(formatted)) == len(values):
            return formatted
    return [f"{v:.10f}" for v in values]

@beartype
def annotate_frame(frame: np.ndarray, predictions: np.ndarray, labels: list[str]) -> Optional[np.ndarray]:
    probs = predictions[0]
    best_idx = int(np.argmax(probs))

    if len(labels) != len(probs):
        console.print(
            f"[bold red]❌ Label count ({len(labels)}) does not match number of prediction probabilities ({len(probs)}).[/bold red]",
        )
        console.print("[yellow]Make sure the number of labels in your script is correct.[/yellow]")
        sys.exit(0)

    formatted_probs = _format_probabilities(probs)

    try:
        # Font automatisch aus Systemverzeichnissen auswählen
        font_path_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",    # Linux
            "C:/Windows/Fonts/arial.ttf",                              # Windows
            "/System/Library/Fonts/Supplemental/Arial.ttf",            # macOS
        ]
        font_path = next((fp for fp in font_path_candidates if os.path.exists(fp)), None)
        if font_path is None:
            raise FileNotFoundError("Kein gültiger TrueType-Font (Arial/DejaVuSans) gefunden.")

        font_size = 20
        outline_width = 2

        # OpenCV-Bild (BGR → RGB → PIL)
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # pylint: disable=no-member
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(font_path, font_size)

        for i, label in enumerate(labels):
            text = f"{label}: {formatted_probs[i]}"
            y = 30 * (i + 1)

            # Farben wie im Original
            fill_color = (0, 255, 0) if i == best_idx else (255, 0, 0)
            outline_color = (0, 0, 0)

            # Text mit Outline (Rand)
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((10 + dx, y + dy), text, font=font, fill=outline_color)

            # Haupttext
            draw.text((10, y), text, font=font, fill=fill_color)

        # Zurück in OpenCV (RGB → BGR)
        frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)  # pylint: disable=no-member

    except (OSError, FileNotFoundError, ValueError, AttributeError, TypeError) as specific_err:
        print("Spezifischer Fehler beim Zeichnen mit TrueType-Font:", specific_err)
        traceback.print_exc()

        # Fallback mit cv2.putText
        for i, label in enumerate(labels):
            text = f"{label}: {formatted_probs[i]}"
            colour = (0, 255, 0) if i == best_idx else (255, 0, 0)
            cv2.putText(  # pylint: disable=no-member
                frame,
                text,
                (10, 30 * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
                0.8,
                colour,
                2,
            )

    return frame

@beartype
def get_shape(filename: Union[str, Path]) -> Optional[list[int]]:
    path = Path(filename)
    if not path.exists():
        console.print(f"[red]Error:[/] File does not exist: {path}")
        return None
    if not path.is_file():
        console.print(f"[red]Error:[/] Path is not a file: {path}")
        return None

    try:
        with console.status(f"Reading shape from file: {path}", spinner="dots"):
            with path.open(mode="r", encoding="utf-8") as f:
                first_line = f.readline()
            match = re.search(r"shape:\s*\((.*)\)", first_line)
            if not match:
                console.print(f"[yellow]Warning:[/] 'shape:' pattern not found in first line of {path}")
                return None
            # safe eval: convert tuple string like "3, 224, 224" to list of ints
            shape_str = match.group(1)
            shape_list = [int(x.strip()) for x in shape_str.split(",") if x.strip().isdigit()]
            if not shape_list:
                console.print(f"[yellow]Warning:[/] No valid integers found in shape in {path}")
                return None
            return shape_list
    except FileNotFoundError as e:
        console.print(f"[red]File not found error for file {path}:[/] {e}")
        return None
    except UnicodeDecodeError as e:
        console.print(f"[red]Encoding error reading file {path}:[/] {e}")
        return None
    except re.error as e:
        console.print(f"[red]Regex error processing file {path}:[/] {e}")
        return None
    except ValueError as e:
        console.print(f"[red]Value conversion error in file {path}:[/] {e}")
        return None
    except IOError as e:
        console.print(f"[red]I/O error reading file {path}:[/] {e}")
        return None


@beartype
def _exit_with_error(msg: str) -> None:
    console.print(f"[bold red]ERROR:[/bold red] {msg}")
    sys.exit(1)

@beartype
def _is_float_list(lst) -> bool:
    try:
        any(float(x) for x in lst)
        return True
    except ValueError:
        return False

@beartype
def _convert_to_ndarray(values: list[str], expected_shape: Any) -> np.ndarray:
    float_values = list(map(float, values))  # Convert strings to floats
    arr = np.array(float_values).reshape(expected_shape)  # Convert to ndarray and reshape
    return arr

# pylint: disable=too-many-branches
@beartype
def load_or_input_model_data(model: Any, filename: str) -> np.ndarray:
    input_shape = model.input_shape  # e.g. (None, 5, 10)
    if input_shape[0] is None:
        expected_shape = input_shape[1:]
    else:
        expected_shape = input_shape

    if os.path.isfile(filename):
        try:
            data = np.loadtxt(filename)
        except FileNotFoundError:
            _exit_with_error(f"File '{filename}' not found.")
        except IsADirectoryError:
            _exit_with_error(f"Expected a file but found a directory: '{filename}'.")
        except ValueError as e:
            _exit_with_error(f"Data format error in '{filename}': {e}")
        except OSError as e:
            _exit_with_error(f"I/O error while reading '{filename}': {e}")

        expected_size = np.prod(expected_shape)
        if data.size != expected_size:
            _exit_with_error(
                f"Data size mismatch. File contains {data.size} elements, "
                f"but model expects input size {expected_size}."
            )

        try:
            data = data.reshape(expected_shape)
        except ValueError as e:
            _exit_with_error(f"Cannot reshape data to {expected_shape}: {e}")
        except TypeError as e:
            _exit_with_error(f"Invalid shape argument {expected_shape}: {e}")

        if not np.issubdtype(data.dtype, np.floating):
            _exit_with_error(f"Data type is not float, but {data.dtype}.")

        return data

    total_values = np.prod(expected_shape)

    while True:
        console.print(f"Please enter {total_values} float values separated by spaces:")
        try:
            user_input = input().strip()
        except KeyboardInterrupt:
            console.print("[yellow]You cancelled with CTRL C[/yellow]")
            sys.exit(1)

        except ValueError as e:
            console.print(f"[red]Failed to convert or reshape manual input to {expected_shape}: {e}. Please try again.[/red]")
            continue
        except TypeError as e:
            console.print(f"[red]Invalid shape argument {expected_shape}: {e}. Please try again.[/red]")
            continue

        values = user_input.split()

        if len(values) != total_values:
            console.print(f"[red]Incorrect number of values entered ({len(values)}), expected {total_values}. Please try again.[/red]")
            continue

        if not _is_float_list(values):
            console.print("[red]Input contains non-float values. Please try again.[/red]")
            continue

        try:
            return _convert_to_ndarray(values, expected_shape)
        except ValueError as e:
            console.print(f"[red]Error trying to convert to Numpy-Array: {e}[/red]")
            continue

@beartype
def show_result(msg) -> None:
    pprint.pprint(msg)
