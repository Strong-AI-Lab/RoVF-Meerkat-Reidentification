"""Cross-platform dependency installer for RoVF.

PyTorch and TorchVision are intentionally installed outside this script because
the correct wheel depends on the user's OS, Python version, CUDA driver, and
hardware.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXTRAS = "models,dev"
VALID_EXTRA_GROUPS = ("models", "segmentation", "dev", "all")
REQUIREMENT_FILES = {
    "core": "requirements-core.txt",
    "models": "requirements-models.txt",
    "segmentation": "requirements-segmentation.txt",
    "dev": "requirements-dev.txt",
}

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class InstallTarget:
    """Commands used to interact with the selected Python environment."""

    mode: str
    target: str
    python_cmd: list[str]
    pip_cmd: list[str]


def normalize_extras(value: str) -> str:
    """Normalize an extras string while preserving group order."""

    return value.replace(" ", "").replace(";", ",")


def add_group(groups: list[str], group: str) -> None:
    if group not in groups:
        groups.append(group)


def build_requirement_groups(minimal: bool, extras: str | None) -> list[str]:
    groups: list[str] = []
    add_group(groups, "core")

    if minimal:
        return groups

    selected_extras = normalize_extras(extras or DEFAULT_EXTRAS)
    if not selected_extras:
        raise ValueError("--extras requires a non-empty comma-separated list.")

    for group in selected_extras.split(","):
        if group == "all":
            add_group(groups, "models")
            add_group(groups, "segmentation")
            add_group(groups, "dev")
        elif group in VALID_EXTRA_GROUPS:
            add_group(groups, group)
        elif not group:
            raise ValueError(f"empty extras group in '{selected_extras}'.")
        else:
            raise ValueError(
                f"unknown extras group '{group}'. "
                "Valid groups: models, segmentation, dev, all"
            )

    return groups


def requirement_file_for_group(group: str, root: Path = SCRIPT_DIR) -> Path:
    return root / REQUIREMENT_FILES[group]


def validate_requirement_files(groups: Iterable[str], root: Path = SCRIPT_DIR) -> None:
    for group in groups:
        req_file = requirement_file_for_group(group, root)
        if not req_file.is_file():
            raise FileNotFoundError(
                f"missing requirements file for group '{group}': {req_file}"
            )


def iter_requirement_lines(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            yield line


def resolve_venv_python(venv_path: Path) -> Path | None:
    candidates = (
        venv_path / "Scripts" / "python.exe",
        venv_path / "Scripts" / "python",
        venv_path / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_conda_commands(
    target: str, conda_executable: str = "conda"
) -> tuple[list[str], list[str]]:
    target_path = Path(target)
    if target_path.is_dir():
        python_cmd = [conda_executable, "run", "--prefix", str(target_path), "python"]
        pip_cmd = [
            conda_executable,
            "run",
            "--prefix",
            str(target_path),
            "python",
            "-m",
            "pip",
        ]
    else:
        python_cmd = [conda_executable, "run", "-n", target, "python"]
        pip_cmd = [conda_executable, "run", "-n", target, "python", "-m", "pip"]

    return python_cmd, pip_cmd


def find_conda_executable(env: dict[str, str] | None = None) -> str | None:
    env = env or os.environ
    conda_exe = env.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).is_file():
        return conda_exe
    return shutil.which("conda")


def detect_install_target(
    mode: str | None,
    target: str | None,
    env: dict[str, str] | None = None,
    root: Path = SCRIPT_DIR,
) -> InstallTarget:
    env = env or os.environ

    if mode is None:
        if env.get("CONDA_PREFIX"):
            mode = "conda"
            target = env["CONDA_PREFIX"]
            print(f"Detected active conda environment: {target}")
        elif env.get("VIRTUAL_ENV"):
            mode = "venv"
            target = env["VIRTUAL_ENV"]
            print(f"Detected active virtual environment: {target}")
        elif resolve_venv_python(root / ".venv") is not None:
            mode = "venv"
            target = str(root / ".venv")
            print(f"Detected local virtual environment: {target}")
        else:
            raise RuntimeError(
                "no environment detected.\n"
                "Provide one explicitly:\n"
                "  python install_packages.py --conda <env_name_or_prefix>\n"
                "  python install_packages.py --venv <venv_path>"
            )

    if mode == "conda":
        if not target:
            raise RuntimeError("--conda requires an environment name or prefix path.")
        conda_executable = find_conda_executable(env)
        if not conda_executable:
            raise RuntimeError("conda not found in PATH.")
        python_cmd, pip_cmd = build_conda_commands(target, conda_executable)
        if Path(target).is_dir():
            print(f"Using conda env prefix: {target}")
        else:
            print(f"Using conda env name: {target}")
        return InstallTarget("conda", target, python_cmd, pip_cmd)

    if mode == "venv":
        if not target:
            raise RuntimeError("--venv requires a path (for example, .venv).")
        python_path = resolve_venv_python(Path(target))
        if python_path is None:
            raise RuntimeError(
                f"could not find python executable under {target}. "
                "Expected Scripts/python.exe, Scripts/python, or bin/python."
            )
        print(f"Using virtual environment at: {target}")
        python_cmd = [str(python_path)]
        pip_cmd = [str(python_path), "-m", "pip"]
        return InstallTarget("venv", target, python_cmd, pip_cmd)

    raise RuntimeError(f"unsupported mode '{mode}'")


def check_pytorch_installed(python_cmd: Sequence[str], runner: Runner = subprocess.run) -> bool:
    result = runner(
        [
            *python_cmd,
            "-c",
            "import torch; import torchvision",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return True

    detail = (result.stderr or result.stdout or "").strip()
    if detail:
        print(detail)
    print("Error: torch and torchvision must be installed before running this script.")
    print("Install the correct build for your platform from:")
    print("  https://pytorch.org/get-started/locally/")
    return False


def print_dry_run(
    groups: Sequence[str],
    constraints_file: Path | None = None,
    root: Path = SCRIPT_DIR,
) -> None:
    if constraints_file is not None:
        print(f"Using constraints file: {constraints_file}")

    print("Dry run only; no packages will be installed.")
    for group in groups:
        req_file = requirement_file_for_group(group, root)
        print()
        print(f"[{group}] {req_file}")
        for line in iter_requirement_lines(req_file):
            print(line)


def install_requirement_groups(
    groups: Sequence[str],
    pip_cmd: Sequence[str],
    constraints_file: Path | None = None,
    runner: Runner = subprocess.run,
    root: Path = SCRIPT_DIR,
) -> int:
    optional_failures: list[str] = []

    for group in groups:
        req_file = requirement_file_for_group(group, root)
        print(f"Installing {group} requirements from {req_file}")

        pip_install_args = [*pip_cmd, "install"]
        if constraints_file is not None:
            pip_install_args.extend(["-c", str(constraints_file)])
        pip_install_args.extend(["-r", str(req_file)])

        result = runner(pip_install_args)
        if result.returncode == 0:
            continue

        if group == "core":
            print("Error: core dependency installation failed.")
            return result.returncode

        print(f"Warning: optional dependency group '{group}' failed to install.")
        optional_failures.append(group)

    if optional_failures:
        print()
        print("Completed with optional dependency warnings.")
        print("The following optional groups failed:")
        for group in optional_failures:
            print(f"  - {group}")
        print("Core dependencies were installed successfully.")
    else:
        print("All selected dependency groups installed successfully.")

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install RoVF dependency groups into conda or venv environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "PyTorch and TorchVision are intentionally not installed here because "
            "the correct wheel depends on your platform/CUDA setup.\n"
            "Install them first from: https://pytorch.org/get-started/locally/"
        ),
    )
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument("--conda", metavar="ENV_OR_PREFIX")
    env_group.add_argument("--venv", metavar="VENV_PATH")
    parser.add_argument(
        "--extras",
        metavar="LIST",
        help="Optional groups to install: models, segmentation, dev, all. Defaults to models,dev.",
    )
    parser.add_argument("--minimal", action="store_true", help="Install core requirements only.")
    constraints_group = parser.add_mutually_exclusive_group()
    constraints_group.add_argument(
        "--reproducible",
        action="store_true",
        help="Install with constraints-validated.txt for repeatable reruns.",
    )
    constraints_group.add_argument(
        "--constraints",
        metavar="PATH",
        help="Install with a caller-provided pip constraints file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print packages without installing.")

    args = parser.parse_args(argv)
    if args.minimal and args.extras is not None:
        parser.error("--minimal cannot be combined with --extras.")

    return args


def main(argv: Sequence[str] | None = None, runner: Runner = subprocess.run) -> int:
    args = parse_args(argv)

    mode = "conda" if args.conda else "venv" if args.venv else None
    target = args.conda or args.venv

    try:
        groups = build_requirement_groups(args.minimal, args.extras)
        validate_requirement_files(groups)

        constraints_file: Path | None = None
        if args.reproducible:
            constraints_file = SCRIPT_DIR / "constraints-validated.txt"
        elif args.constraints:
            constraints_file = Path(args.constraints)

        if constraints_file is not None and not constraints_file.is_file():
            raise FileNotFoundError(f"constraints file not found: {constraints_file}")

        print(f"Selected requirement groups: {' '.join(groups)}")

        if args.dry_run:
            print_dry_run(groups, constraints_file)
            return 0

        install_target = detect_install_target(mode, target)
        if not check_pytorch_installed(install_target.python_cmd, runner):
            return 1

        return install_requirement_groups(
            groups,
            install_target.pip_cmd,
            constraints_file=constraints_file,
            runner=runner,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
