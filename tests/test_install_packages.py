import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import install_packages as installer


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "install_packages.py"
BASH_WRAPPER = REPO_ROOT / "install_packages.sh"
POWERSHELL_WRAPPER = REPO_ROOT / "install_packages.ps1"


def _run_installer(args, **kwargs):
    return subprocess.run(
        [sys.executable, str(INSTALLER), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        **kwargs,
    )


def test_install_cli_help_succeeds():
    result = _run_installer(["--help"])

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--reproducible" in result.stdout
    assert "--constraints" in result.stdout


def test_install_cli_dry_run_uses_default_groups_without_environment():
    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env.pop("VIRTUAL_ENV", None)

    result = _run_installer(["--dry-run"], env=env)

    assert result.returncode == 0
    assert "Selected requirement groups: core models dev" in result.stdout
    assert "Using constraints file" not in result.stdout
    assert "requirements-core.txt" in result.stdout
    assert "requirements-models.txt" in result.stdout
    assert "requirements-dev.txt" in result.stdout
    assert "requirements-segmentation.txt" not in result.stdout


def test_install_cli_reproducible_dry_run_uses_validated_constraints():
    result = _run_installer(["--reproducible", "--dry-run"])

    assert result.returncode == 0
    assert "Using constraints file:" in result.stdout
    assert "constraints-validated.txt" in result.stdout


def test_install_cli_custom_constraints_dry_run(tmp_path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("numpy==1.26.4\n", encoding="utf-8")

    result = _run_installer(["--constraints", str(constraints), "--dry-run"])

    assert result.returncode == 0
    assert f"Using constraints file: {constraints}" in result.stdout


def test_install_cli_missing_custom_constraints_fails(tmp_path):
    missing_constraints = tmp_path / "missing-constraints.txt"

    result = _run_installer(["--constraints", str(missing_constraints), "--dry-run"])

    assert result.returncode != 0
    assert "constraints file not found" in result.stdout


def test_install_cli_minimal_dry_run_uses_core_only():
    result = _run_installer(["--minimal", "--dry-run"])

    assert result.returncode == 0
    assert "Selected requirement groups: core" in result.stdout
    assert "requirements-core.txt" in result.stdout
    assert "requirements-models.txt" not in result.stdout


def test_install_cli_all_extras_dry_run_includes_all_groups():
    result = _run_installer(["--extras", "all", "--dry-run"])

    assert result.returncode == 0
    assert "Selected requirement groups: core models segmentation dev" in result.stdout
    assert "requirements-core.txt" in result.stdout
    assert "requirements-models.txt" in result.stdout
    assert "requirements-segmentation.txt" in result.stdout
    assert "requirements-dev.txt" in result.stdout


def test_install_cli_minimal_conflicts_with_extras():
    result = _run_installer(["--minimal", "--extras", "models", "--dry-run"])

    assert result.returncode != 0
    assert "--minimal cannot be combined with --extras" in result.stderr


def test_install_cli_unknown_extra_fails():
    result = _run_installer(["--extras", "models,unknown", "--dry-run"])

    assert result.returncode != 0
    assert "unknown extras group 'unknown'" in result.stdout


def test_normalize_extras_accepts_spaces_and_semicolons():
    assert installer.normalize_extras("models; segmentation, dev") == "models,segmentation,dev"


def test_requirement_group_selection_preserves_order_and_de_duplicates():
    groups = installer.build_requirement_groups(False, "dev,models,dev,all")

    assert groups == ["core", "dev", "models", "segmentation"]


def test_resolve_venv_python_supports_windows_layout(tmp_path):
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert installer.resolve_venv_python(tmp_path / "venv") == python


def test_resolve_venv_python_supports_posix_layout(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert installer.resolve_venv_python(tmp_path / "venv") == python


def test_detect_install_target_auto_detects_local_posix_venv(tmp_path, capsys):
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    target = installer.detect_install_target(
        None,
        None,
        env={"PATH": ""},
        root=tmp_path,
    )

    captured = capsys.readouterr()
    assert target.mode == "venv"
    assert target.target == str(tmp_path / ".venv")
    assert target.python_cmd == [str(python)]
    assert target.pip_cmd == [str(python), "-m", "pip"]
    assert "Detected local virtual environment" in captured.out
    assert "Using virtual environment at" in captured.out


def test_build_conda_commands_for_name():
    python_cmd, pip_cmd = installer.build_conda_commands("rovf", "conda")

    assert python_cmd == ["conda", "run", "-n", "rovf", "python"]
    assert pip_cmd == ["conda", "run", "-n", "rovf", "python", "-m", "pip"]


def test_build_conda_commands_for_prefix(tmp_path):
    env_prefix = tmp_path / "envs" / "rovf"
    env_prefix.mkdir(parents=True)

    python_cmd, pip_cmd = installer.build_conda_commands(str(env_prefix), "conda")

    assert python_cmd == ["conda", "run", "--prefix", str(env_prefix), "python"]
    assert pip_cmd == [
        "conda",
        "run",
        "--prefix",
        str(env_prefix),
        "python",
        "-m",
        "pip",
    ]


def test_install_groups_warns_for_optional_group_failure(capsys):
    def fake_runner(command, **kwargs):
        if any(str(part).endswith("requirements-models.txt") for part in command):
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    result = installer.install_requirement_groups(
        ["core", "models"],
        [sys.executable, "-m", "pip"],
        runner=fake_runner,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Warning: optional dependency group 'models' failed to install" in captured.out
    assert "Core dependencies were installed successfully" in captured.out


def test_install_groups_passes_constraints_to_pip(tmp_path):
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("numpy==1.26.4\n", encoding="utf-8")
    calls = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    result = installer.install_requirement_groups(
        ["core"],
        [sys.executable, "-m", "pip"],
        constraints_file=constraints,
        runner=fake_runner,
    )

    assert result == 0
    assert "-c" in calls[0]
    assert str(constraints) in calls[0]
    assert any(str(part).endswith("requirements-core.txt") for part in calls[0])


def test_main_with_posix_venv_uses_python_m_pip_without_installing(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    result = installer.main(
        ["--venv", str(tmp_path / "venv"), "--minimal"],
        runner=fake_runner,
    )

    assert result == 0
    assert len(calls) == 2
    assert calls[0][0] == [str(python), "-c", "import torch; import torchvision"]
    assert calls[0][1]["text"] is True
    assert calls[0][1]["capture_output"] is True
    assert calls[1][0][:4] == [str(python), "-m", "pip", "install"]
    assert any(str(part).endswith("requirements-core.txt") for part in calls[1][0])
    assert all(
        not any(str(part).endswith(name) for part in calls[1][0])
        for name in (
            "requirements-models.txt",
            "requirements-segmentation.txt",
            "requirements-dev.txt",
        )
    )


def test_bash_wrapper_dry_run_smoke():
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available")

    result = subprocess.run(
        [bash, str(BASH_WRAPPER), "--dry-run"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Selected requirement groups: core models dev" in result.stdout


def test_bash_wrapper_preserves_constraint_path_with_spaces(tmp_path):
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is not available")

    constraints = tmp_path / "constraints with spaces.txt"
    constraints.write_text("numpy==1.26.4\n", encoding="utf-8")

    result = subprocess.run(
        [bash, str(BASH_WRAPPER), "--constraints", str(constraints), "--dry-run"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert f"Using constraints file: {constraints}" in result.stdout
    assert "Selected requirement groups: core models dev" in result.stdout


def test_powershell_wrapper_dry_run_smoke():
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is not available")

    command = [powershell, "-NoProfile"]
    if os.name == "nt":
        command.extend(["-ExecutionPolicy", "Bypass"])
    command.extend(["-File", str(POWERSHELL_WRAPPER), "--dry-run"])

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Selected requirement groups: core models dev" in result.stdout
