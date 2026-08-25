import importlib

from seismic_hydro_ml import config


def test_default_data_dirs_are_under_repo_root():
    assert config.INPUT_DIR == config.REPO_ROOT / "input"
    assert config.MODELS_DIR == config.REPO_ROOT / "models"
    assert config.OUTPUT_DIR == config.REPO_ROOT / "output"


def test_env_var_overrides_input_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SHML_INPUT_DIR", str(tmp_path))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.INPUT_DIR == tmp_path
    finally:
        monkeypatch.delenv("SHML_INPUT_DIR", raising=False)
        importlib.reload(config)


def test_raw_data_dirs_default_to_none_without_env(monkeypatch):
    monkeypatch.delenv("SHML_RAW_VR_DIR", raising=False)
    monkeypatch.delenv("SHML_RAW_PIEZO_DIR", raising=False)
    reloaded = importlib.reload(config)
    try:
        assert reloaded.RAW_VR_DIR is None
        assert reloaded.RAW_PIEZO_DIR is None
    finally:
        importlib.reload(config)


def test_ensure_data_dirs_creates_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("SHML_INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("SHML_MODELS_DIR", str(tmp_path / "mdl"))
    monkeypatch.setenv("SHML_OUTPUT_DIR", str(tmp_path / "out"))
    reloaded = importlib.reload(config)
    try:
        reloaded.ensure_data_dirs()
        assert (tmp_path / "in").is_dir()
        assert (tmp_path / "mdl").is_dir()
        assert (tmp_path / "out").is_dir()
    finally:
        for var in ("SHML_INPUT_DIR", "SHML_MODELS_DIR", "SHML_OUTPUT_DIR"):
            monkeypatch.delenv(var, raising=False)
        importlib.reload(config)
