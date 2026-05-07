import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "install_packages.sh"


def _run_install(args, **kwargs):
    return subprocess.run(
        [str(INSTALL_SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        **kwargs,
    )


def test_install_script_help_succeeds():
    result = _run_install(["--help"])

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "--reproducible" in result.stdout
    assert "--constraints <path>" in result.stdout


def test_install_script_dry_run_uses_default_groups():
    result = _run_install(["--dry-run"])

    assert result.returncode == 0
    assert "Selected requirement groups: core models dev" in result.stdout
    assert "Using constraints file" not in result.stdout
    assert "requirements-core.txt" in result.stdout
    assert "requirements-models.txt" in result.stdout
    assert "requirements-dev.txt" in result.stdout
    assert "requirements-segmentation.txt" not in result.stdout


def test_install_script_reproducible_dry_run_uses_validated_constraints():
    result = _run_install(["--reproducible", "--dry-run"])

    assert result.returncode == 0
    assert "Using constraints file:" in result.stdout
    assert "constraints-validated.txt" in result.stdout


def test_install_script_missing_custom_constraints_fails(tmp_path):
    missing_constraints = tmp_path / "missing-constraints.txt"

    result = _run_install(["--constraints", str(missing_constraints), "--dry-run"])

    assert result.returncode != 0
    assert "constraints file not found" in result.stdout


def test_install_script_minimal_dry_run_uses_core_only():
    result = _run_install(["--minimal", "--dry-run"])

    assert result.returncode == 0
    assert "Selected requirement groups: core" in result.stdout
    assert "requirements-core.txt" in result.stdout
    assert "requirements-models.txt" not in result.stdout


def test_install_script_all_extras_dry_run_includes_all_groups():
    result = _run_install(["--extras", "all", "--dry-run"])

    assert result.returncode == 0
    assert "Selected requirement groups: core models segmentation dev" in result.stdout
    assert "requirements-core.txt" in result.stdout
    assert "requirements-models.txt" in result.stdout
    assert "requirements-segmentation.txt" in result.stdout
    assert "requirements-dev.txt" in result.stdout


def test_install_script_unknown_argument_fails():
    result = _run_install(["--not-a-real-option"])

    assert result.returncode != 0
    assert "Unknown argument" in result.stdout


def test_install_script_missing_venv_path_fails():
    result = _run_install(["--venv", "/tmp/rovf-missing-venv"])

    assert result.returncode != 0
    assert "could not find python executable" in result.stdout


def test_install_script_no_environment_detected_fails(tmp_path):
    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(
        [str(INSTALL_SCRIPT)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert "no environment detected" in result.stdout


def test_install_script_warns_for_optional_group_failure(tmp_path):
    fake_venv = tmp_path / "venv"
    bin_dir = fake_venv / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(
        """#!/bin/bash
if [[ "$1" == "-m" && "$2" == "pip" ]]; then
  req_file="${@: -1}"
  if [[ "$req_file" == *"requirements-models.txt" ]]; then
    exit 1
  fi
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    python.chmod(0o755)

    result = _run_install(["--venv", str(fake_venv), "--extras", "models"])

    assert result.returncode == 0
    assert "Warning: optional dependency group 'models' failed to install" in result.stdout
    assert "Core dependencies were installed successfully" in result.stdout


def test_install_script_passes_constraints_to_pip(tmp_path):
    fake_venv = tmp_path / "venv"
    bin_dir = fake_venv / "bin"
    bin_dir.mkdir(parents=True)
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("numpy==1.26.4\n", encoding="utf-8")
    pip_args_log = tmp_path / "pip-args.txt"
    python = bin_dir / "python"
    python.write_text(
        f"""#!/bin/bash
if [[ "$1" == "-m" && "$2" == "pip" ]]; then
  printf '%s\\n' "$@" >> "{pip_args_log}"
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    python.chmod(0o755)

    result = _run_install(
        ["--venv", str(fake_venv), "--minimal", "--constraints", str(constraints)]
    )

    assert result.returncode == 0
    pip_args = pip_args_log.read_text(encoding="utf-8")
    assert "-c" in pip_args
    assert str(constraints) in pip_args
    assert "requirements-core.txt" in pip_args
