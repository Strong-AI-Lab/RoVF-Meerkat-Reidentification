import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

FIXED_HYP_YAMLS = (
    "50_50_0p5_fps_aug.yml",
    "50_50_0p5_fps_no_aug.yml",
    "50_50_1_fps_aug.yml",
    "50_50_1_fps_no_aug.yml",
    "mask_0p5_fps_aug.yml",
    "mask_0p5_fps_no_aug.yml",
    "mask_1_fps_aug.yml",
    "mask_1_fps_no_aug.yml",
    "no_mask_0p5_fps_aug.yml",
    "no_mask_0p5_fps_no_aug.yml",
    "no_mask_1_fps_aug.yml",
    "no_mask_1_fps_no_aug.yml",
)

VID_OTHER_YAMLS = (
    "50_50_aug.yml",
    "50_50_no_aug.yml",
    "mask_aug.yml",
    "mask_no_aug.yml",
    "no_mask_aug.yml",
    "no_mask_no_aug.yml",
)


@pytest.fixture
def bash():
    bash_path = shutil.which("bash")
    if not bash_path:
        pytest.skip("bash is not available")
    return bash_path


def _run_script(bash, relative_path, args=(), cwd=None, env=None):
    return subprocess.run(
        [bash, str(REPO_ROOT / relative_path), *map(str, args)],
        cwd=cwd or REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def _make_base(tmp_path, with_dataset=True):
    base = tmp_path / "base dir"
    _touch(base / "main.py")
    _touch(base / "evaluation" / "get_embeddings.py")

    if with_dataset:
        _touch(base / "Dataset" / "meerkat_h5files" / "Precomputed_test_examples_meerkat.csv")
        _touch(base / "Dataset" / "meerkat_h5files" / "Cooccurrences.json")
        _touch(base / "Dataset" / "meerkat_h5files" / "masks" / "meerkat_masks.pkl")
        (base / "Dataset" / "meerkat_h5files" / "clips" / "Test").mkdir(parents=True)

        _touch(base / "Dataset" / "polarbears_h5files" / "Precomputed_test_examples_polarbear.csv")
        _touch(base / "Dataset" / "polarbears_h5files" / "Cooccurrences.json")
        _touch(base / "Dataset" / "polarbears_h5files" / "masks" / "PB_masks.pkl")
        (base / "Dataset" / "polarbears_h5files" / "clips" / "Test").mkdir(parents=True)

    return base


def _make_fake_python(tmp_path):
    fake_python = tmp_path / "fake python"
    log_path = tmp_path / "python.log"
    fake_python.write_text(
        """#!/usr/bin/env bash
{
  printf 'PWD=%s\\n' "$PWD"
  printf 'CUDA=%s\\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'ARGS'
  for arg in "$@"; do
    printf '\\t%s' "$arg"
  done
  printf '\\n'
} >> "$PYTHON_LOG"
exit "${FAKE_PYTHON_EXIT:-0}"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["PYTHON_LOG"] = str(log_path)
    return fake_python, log_path, env


def _create_yamls(base, model_name, names):
    yaml_dir = base / "training_scripts" / "exp_metadata" / "hyperparameter_search" / model_name
    for name in names:
        _touch(yaml_dir / name)
    return yaml_dir


def _commands_from_log(log_path):
    if not log_path.exists():
        return []
    return [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ARGS")
    ]


def test_all_shell_scripts_parse_with_bash_n(bash):
    for script in sorted(REPO_ROOT.rglob("*.sh")):
        result = subprocess.run(
            [bash, "-n", str(script)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, f"{script} failed bash -n:\n{result.stderr}"


def test_gitattributes_forces_lf_for_shell_scripts():
    assert "*.sh text eol=lf" in (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")


def test_fixed_hyp_search_keeps_legacy_positional_gpu_from_other_cwd(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    _create_yamls(base, "LSTM", FIXED_HYP_YAMLS)
    fake_python, log_path, env = _make_fake_python(tmp_path)
    cwd = tmp_path / "cwd with spaces"
    cwd.mkdir()

    result = _run_script(
        bash,
        "run_hyp_search.sh",
        ["--base-dir", base, "--python-bin", fake_python, "LSTM", "7"],
        cwd=cwd,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    log = log_path.read_text(encoding="utf-8")
    assert log.count("ARGS\t") == len(FIXED_HYP_YAMLS)
    assert log.count("CUDA=7") == len(FIXED_HYP_YAMLS)
    assert f"PWD={base}" in log
    assert "training_scripts/exp_metadata/hyperparameter_search/LSTM/50_50_0p5_fps_aug.yml" in log


def test_fixed_hyp_search_fails_when_yaml_is_missing(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    _create_yamls(base, "LSTM", FIXED_HYP_YAMLS[:1])
    fake_python, _, env = _make_fake_python(tmp_path)

    result = _run_script(
        bash,
        "run_hyp_search.sh",
        ["--base-dir", base, "--python-bin", fake_python, "LSTM", "0"],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode != 0
    assert "YAML config not found" in result.stderr


def test_video_other_search_supports_cpu_dry_run_without_gpu(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    _create_yamls(base, "timesformer", VID_OTHER_YAMLS)

    result = _run_script(
        bash,
        "run_hyp_search_vid_other.sh",
        ["--base-dir", base, "--device", "cpu", "--dry-run", "timesformer"],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("DRY_RUN:") == len(VID_OTHER_YAMLS)
    assert "-d cpu" in result.stdout


def test_generated_hyperparameter_search_continues_but_returns_nonzero_on_failures(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    yaml_dir = _create_yamls(base, "bioclip_video", ("b.yml", "a.yml"))
    fake_python, log_path, env = _make_fake_python(tmp_path)
    env["FAKE_PYTHON_EXIT"] = "3"

    result = _run_script(
        bash,
        "run_bioclip_video_hyperparameter_search.sh",
        ["--base-dir", base, "--yaml-dir", yaml_dir, "--python-bin", fake_python, "--gpu", "5"],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode != 0
    assert result.stderr.count("Training failed for:") == 2
    log = log_path.read_text(encoding="utf-8")
    assert log.count("ARGS\t") == 2
    assert log.count("CUDA=5") == 2
    assert log.find("a.yml") < log.find("b.yml")


def test_checkpoint_test_runner_fails_for_missing_model_directory(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    fake_python, _, env = _make_fake_python(tmp_path)

    result = _run_script(
        bash,
        "run_bioclip_video_tests.sh",
        ["--base-dir", base, "--python-bin", fake_python],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode != 0
    assert "model directory not found" in result.stderr


def test_checkpoint_test_runner_runs_mask_and_nomask_with_space_paths(bash, tmp_path):
    base = _make_base(tmp_path)
    model_dir = tmp_path / "model dir"
    checkpoint = _touch(model_dir / "nested" / "checkpoint_epoch_1 with spaces.pt")
    fake_python, log_path, env = _make_fake_python(tmp_path)

    result = _run_script(
        bash,
        "run_bioclip_video_tests.sh",
        ["--base-dir", base, "--model-dir", model_dir, "--python-bin", fake_python, "--gpu", "2"],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    commands = _commands_from_log(log_path)
    assert len(commands) == 2
    assert str(checkpoint) in commands[0]
    assert "\t-m\t" in commands[0]
    assert "\t-m\t" not in commands[1]
    assert log_path.read_text(encoding="utf-8").count("CUDA=2") == 2


def test_interactive_checkpoint_script_requires_flags_without_tty(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    fake_python, _, env = _make_fake_python(tmp_path)

    result = _run_script(
        bash,
        "get_emb_and_metric.sh",
        ["--base-dir", base, "--python-bin", fake_python],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode != 0
    assert "--animal is required" in result.stderr


def test_interactive_checkpoint_script_rejects_invalid_animal(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    checkpoint = _touch(tmp_path / "checkpoint.pt")

    result = _run_script(
        bash,
        "get_emb_and_metric.sh",
        ["--base-dir", base, "--animal", "ferret", "--device", "cpu", "--checkpoint", checkpoint, "--dry-run"],
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "invalid animal" in result.stderr


def test_checkpoint_dir_discovers_sorted_checkpoints_and_preserves_spaces(bash, tmp_path):
    base = _make_base(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint dir"
    second = _touch(checkpoint_dir / "b checkpoint.pt")
    first = _touch(checkpoint_dir / "a checkpoint.pt")
    fake_python, log_path, env = _make_fake_python(tmp_path)

    result = _run_script(
        bash,
        "get_emb_and_metric.sh",
        [
            "--base-dir",
            base,
            "--python-bin",
            fake_python,
            "--animal",
            "meerkat",
            "--device",
            "cpu",
            "--checkpoint-dir",
            checkpoint_dir,
        ],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    log = log_path.read_text(encoding="utf-8")
    assert log.count("ARGS\t") == 4
    assert log.find(str(first)) < log.find(str(second))
    assert str(first) in log
    assert str(second) in log


def test_manual_image_majority_script_adds_imv_flag(bash, tmp_path):
    base = _make_base(tmp_path)
    checkpoint = _touch(tmp_path / "checkpoint with spaces.pt")
    fake_python, log_path, env = _make_fake_python(tmp_path)

    result = _run_script(
        bash,
        "get_emb_and_metric_manual_image_maj.sh",
        [
            "--base-dir",
            base,
            "--python-bin",
            fake_python,
            "--animal",
            "meerkat",
            "--device",
            "cpu",
            "--checkpoint",
            checkpoint,
        ],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    commands = _commands_from_log(log_path)
    assert len(commands) == 2
    assert all("\t-imv\tTrue" in command for command in commands)


def test_evaluation_bioclip_dry_run_from_arbitrary_cwd(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    fake_python, _, _ = _make_fake_python(tmp_path)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()

    result = _run_script(
        bash,
        "evaluation/get_bioclip_embeddings.sh",
        ["--base-dir", base, "--python-bin", fake_python, "--dry-run"],
        cwd=cwd,
    )

    assert result.returncode == 0, result.stderr
    assert "./get_embeddings.py" in result.stdout
    assert "--model_type bioclip" in result.stdout
    assert "--load_masks" in result.stdout
    assert "--mask_path none" in result.stdout


def test_evaluation_image_majority_dry_run_includes_image_vote_flag(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    fake_python, _, _ = _make_fake_python(tmp_path)

    result = _run_script(
        bash,
        "evaluation/get_embeddings_image_maj_vot.sh",
        ["--base-dir", base, "--python-bin", fake_python, "--dry-run"],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "--image_maj_vote" in result.stdout


def test_evaluation_non_dry_run_validates_dataset_before_python(bash, tmp_path):
    base = _make_base(tmp_path, with_dataset=False)
    fake_python, log_path, env = _make_fake_python(tmp_path)

    result = _run_script(
        bash,
        "evaluation/get_dino_embeddings.sh",
        ["--base-dir", base, "--python-bin", fake_python],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode != 0
    assert "mask file not found" in result.stderr
    assert not log_path.exists()
