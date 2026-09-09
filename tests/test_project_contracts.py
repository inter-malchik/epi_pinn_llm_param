"""Contracts the rest of the project relies on: dataset schemas, entry-point constants,
notebook import paths, dependency pins.

Reading guide
-------------
* These are characterization tests: they pin the CURRENT behavior of the code,
  quirks included. A comment starting with ``# NOTE: current behavior`` marks a
  place where the pinned behavior looks like a bug. Fix the code first, then
  update the test on purpose - never "fix" the test to hide the change.
* Test names read as sentences (``test_zero_beta_is_treated_as_missing``); a
  test class groups the tests of one function, method or scenario.
* Only external boundaries are faked (LLM providers, the network). The SIRD
  solver, the PINN, LangGraph and the file system are real. Shared fixtures
  live in ``tests/conftest.py``, fake LLM clients and JSON reply builders in
  ``tests/support.py``.
"""
import importlib
import inspect
import json
import re

import pandas as pd
import pytest

from agents.PINN_const import EINN_PINN
from tests.conftest import DATA_DIR, ITALY_CSV, PROJECT_ROOT, REAL_CSV, SYNTHETIC_CSV

MAIN_SOURCE = (PROJECT_ROOT / "main_test.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- datasets


class TestDatasets:
    """Schemas and sizes of the CSV datasets under 'Phase 1 (model calibration)'."""

    def test_synthetic_files(self):
        names = sorted(p.name for p in (DATA_DIR / "synthetic_datasets").glob("*.csv"))
        assert names == [
            "01_baseline_constant.csv",
            "02_lockdown_beta_jump.csv",
            "02_lockdown_beta_jump_noise_10pct_seed456.csv",
            "02_lockdown_beta_jump_noise_3pct_seed42.csv",
            "02_lockdown_beta_jump_noise_5pct_seed123.csv",
            "03_beta_baseline_constant.csv",
            "03_seasonal_beta_sin.csv",
            "04_decaying_beta_trend.csv",
            "05_complex_beta_gamma_dynamics.csv",
            "05_complex_beta_gamma_dynamics_noise_10pct_seed456.csv",
            "05_complex_beta_gamma_dynamics_noise_3pct_seed42.csv",
            "05_complex_beta_gamma_dynamics_noise_5pct_seed123.csv",
        ]

    def test_synthetic_schema(self, synthetic_df):
        assert list(synthetic_df.columns) == ["day", "S", "I", "R", "D", "beta", "gamma", "mu", "R0"]
        assert len(synthetic_df) == 366
        assert synthetic_df["day"].tolist() == list(range(366))

    def test_synthetic_baseline_ground_truth(self, synthetic_df):
        assert synthetic_df["beta"].nunique() == 1 and synthetic_df["beta"].iloc[0] == 0.1
        assert synthetic_df["gamma"].iloc[0] == 0.06 and synthetic_df["mu"].iloc[0] == 0.003
        assert synthetic_df["R0"].iloc[0] == pytest.approx(0.1 / 0.063)
        first = synthetic_df.iloc[0]
        assert (first["S"], first["I"], first["R"], first["D"]) == (999.0, 1.0, 0.0, 0.0)
        total = synthetic_df[["S", "I", "R", "D"]].sum(axis=1)
        assert total.max() == pytest.approx(1000.0, rel=1e-6) and total.min() == pytest.approx(1000.0, rel=1e-6)

    def test_every_synthetic_file_shares_the_schema(self):
        for path in (DATA_DIR / "synthetic_datasets").glob("*.csv"):
            df = pd.read_csv(path)
            assert list(df.columns) == ["day", "S", "I", "R", "D", "beta", "gamma", "mu", "R0"], path.name
            assert len(df) == 366, path.name

    def test_real_st_petersburg_schema(self, real_df):
        assert list(real_df.columns) == ["t", "I", "D", "S", "R"]
        assert len(real_df) == 369
        assert real_df.iloc[0].tolist() == [0.0, 4197, 42, 5995513, 248]

    def test_real_italy_schema(self):
        df = pd.read_csv(ITALY_CSV)
        assert list(df.columns) == ["S", "I", "R", "D"]
        assert len(df) == 403
        assert df.iloc[0].tolist() == [60461826, 0, 0, 0]


# --------------------------------------------------------------------------- main() constants


class TestEntryPointConstants:
    """Hard-coded configuration inside main() that other people rely on (or trip over)."""

    def test_data_path_points_outside_the_repository(self):
        # NOTE: current behavior — possible bug: main() reads ../../NEW_PINN/..., which does
        # not exist in this repository; the same file lives under Phase 1 (model calibration)
        active = re.findall(r"^\s*covid_cases = pd\.read_csv\('([^']+)'\)", MAIN_SOURCE, re.M)
        assert active == ["../../NEW_PINN/synthetic_datasets/01_baseline_constant.csv"]
        commented = re.findall(r"^\s*# covid_cases = pd\.read_csv\('([^']+)'\)", MAIN_SOURCE, re.M)
        assert len(commented) == 3  # the alternative datasets are kept as comments
        assert not (PROJECT_ROOT / "../../NEW_PINN").resolve().exists()
        assert SYNTHETIC_CSV.exists()

    def test_synthetic_mode_and_baseline_parameters(self):
        assert "USE_SYNTHETIC_DATA = True" in MAIN_SOURCE
        assert "baseline_beta = 0.091" in MAIN_SOURCE
        assert "baseline_gamma = 0.0553" in MAIN_SOURCE
        assert "baseline_mu = 0.0085" in MAIN_SOURCE
        assert re.search(r'^\s*expert_comment = "Need higher peak"$', MAIN_SOURCE, re.M)

    def test_training_configuration(self):
        assert re.search(r"^\s*n_epoch = 10000$", MAIN_SOURCE, re.M)
        assert re.search(r"^\s*lambda_data = 1\.0$", MAIN_SOURCE, re.M)
        assert re.search(r"^\s*lambda_ode = 1\.0$", MAIN_SOURCE, re.M)
        assert re.search(r"^\s*lambda_bc = 0\.0$", MAIN_SOURCE, re.M)
        assert "'train_size': 120," in MAIN_SOURCE

    def test_printed_and_passed_max_iterations_disagree(self):
        # NOTE: current behavior — the log says 5, run() receives 10
        assert 'print(f"   Max iterations: 5")' in MAIN_SOURCE
        assert re.search(r"max_iterations=10,\s*\n\s*t_max=400", MAIN_SOURCE)

    def test_verification_node_epochs_are_hard_coded_in_build_graph(self):
        assert "PINNVerificationNode(n_epoch=10000, save_plots=True)" in MAIN_SOURCE

    def test_train_split_scaling_factor(self):
        assert "train_split_time=int(t_train_split*2.5)" in MAIN_SOURCE

    def test_simulate_is_defined_twice_in_sensitivity_node(self):
        # NOTE: current behavior — two identical definitions; the second one wins
        assert MAIN_SOURCE.count("    def _simulate(self, beta, gamma, mu, population, S0, I0, R0, D0, t_max, num_points):") == 2

    def test_expert_comment_examples_file(self):
        text = (PROJECT_ROOT / "expert_comment_peak_examples.txt").read_text(encoding="utf-8")
        for comment in ("Need higher peak", "Need lower peak", "Need earlier peak", "Need later peak"):
            assert f'"{comment}"' in text
        assert "PEAK POSITION ONLY" in text and "PEAK HEIGHT ONLY" in text and "COMBINATIONS" in text


# --------------------------------------------------------------------------- notebooks


def _notebook_sources(name):
    nb = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


class TestNotebookContracts:
    """Import paths, data files and signatures the Phase 1 notebooks depend on."""

    def test_pinn_test_notebook_imports(self):
        cells = _notebook_sources("PINN_test.ipynb")
        assert cells[0].strip() == 'import sys\nsys.path.append("../../")'
        assert "from agents.PINN_const import EINN_PINN" in cells[1]
        # NOTE: current behavior — the λ-sweep cell imports a package that is not in this repo
        assert "from NEW_PINN_LLM_SUR.LLM_epiparam_generator.agents.PINN_const import EINN_PINN" in cells[3]
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("NEW_PINN_LLM_SUR.LLM_epiparam_generator.agents.PINN_const")

    def test_pinn_test_notebook_data_and_parameters(self):
        cells = _notebook_sources("PINN_test.ipynb")
        assert "covid_cases = pd.read_csv('./real_datasets/covid-19_Kouprianov.csv')" in cells[1]
        assert "train_size = 185" in cells[1]
        assert '"beta": 0.0458' in cells[1] and '"gamma": 0.0371' in cells[1] and '"mu": 0.00229' in cells[1]
        assert REAL_CSV.exists()

    def test_einn_pinn_signature_matches_the_notebooks(self):
        params = list(inspect.signature(EINN_PINN.__init__).parameters)
        assert params == ["self", "t", "S_data", "I_data", "R_data", "D_data", "population", "train_size", "init_params", "device"]
        train = inspect.signature(EINN_PINN.train_model).parameters
        assert {k: v.default for k, v in train.items() if k != "self"} == {"n_epoch": 20000, "lambda_data": 1.0, "lambda_ode": 0.1, "lambda_ic": 0.1, "lambda_bc": 0.1}

    def test_sird_calibration_notebook_bounds(self):
        cells = _notebook_sources("SIRD_calibration.ipynb")
        assert "bounds = [(0.01, 0.99), (0.001, 0.5), (0.0001, 0.1)]" in cells[0]
        assert "method: str = 'L-BFGS-B'" in cells[0]
        assert "plot_sird(beta=0.1295, gamma=0.0985, mu=0.00985, N = 6_000_000, I0=4000)" in cells[1]


# --------------------------------------------------------------------------- dependencies


class TestDependencyPins:
    """Facts about requirements.txt that the CI workflow and Makefile work around."""

    def test_requirements_pin_a_cuda_torch_build(self):
        # documents why CI and `make install` strip torch/torchvision from requirements.txt
        req = (PROJECT_ROOT / "requirements.txt").read_text()
        assert "torch==2.8.0+cu128" in req and "torchvision==0.23.0+cu128" in req
        assert "langgraph==1.1.3" in req and "langchain-core==1.2.22" in req and "pydantic==2.11.9" in req

    def test_dev_requirements_do_not_include_runtime_pins(self):
        dev = (PROJECT_ROOT / "requirements-dev.txt").read_text()
        assert "pytest" in dev and "-r requirements.txt" not in dev
