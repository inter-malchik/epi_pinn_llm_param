"""agents/PINN_const.py — EpiParams, TorchStandardScaler and the EINN_PINN network.

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
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.PINN_const import EINN_PINN, EpiParams, TorchStandardScaler

# --------------------------------------------------------------------------- EpiParams


class TestEpiParams:
    """Logit-space parameterization of β, γ, μ: clamping, round-trip and the frozen (requires_grad=False) parameters."""

    def test_defaults(self):
        p = EpiParams(population_n=1000)
        assert p.N == 1000 and p.device == "cpu"
        assert p.beta.item() == pytest.approx(0.3, abs=1e-6)
        assert p.gamma.item() == pytest.approx(0.1, abs=1e-6)
        assert p.mu.item() == pytest.approx(0.04, abs=1e-6)

    def test_init_params_round_trip_through_logit_space(self):
        p = EpiParams(1000, init_params={"beta": 0.091, "gamma": 0.0553, "mu": 0.0085})
        assert p.get_params_dict() == pytest.approx({"beta": 0.091, "gamma": 0.0553, "mu": 0.0085}, abs=1e-6)

    def test_latents_are_frozen_parameters(self):
        # NOTE: current behavior — β, γ, μ are nn.Parameters with requires_grad=False:
        # they take part in the optimizer's param list but never receive a gradient
        p = EpiParams(1000)
        latents = list(p.parameters())
        assert len(latents) == 3
        assert all(isinstance(l, nn.Parameter) for l in latents)
        assert all(l.requires_grad is False for l in latents)
        assert p.beta_latent.dtype == torch.float64

    @pytest.mark.parametrize("value,expected", [(0.0, 0.0), (-0.5, 0.0), (1.0, 1.0), (1.5, 1.0)])
    def test_boundaries_are_clamped_into_the_unit_interval(self, value, expected):
        p = EpiParams(1000, init_params={"beta": value, "gamma": 0.1, "mu": 0.01})
        assert p.beta.item() == pytest.approx(expected, abs=1e-9)
        assert abs(p.beta_latent.item()) > 13  # arctanh(±(1 - 2e-12))

    def test_latent_value_is_arctanh(self):
        p = EpiParams(1000, init_params={"beta": 0.25, "gamma": 0.1, "mu": 0.01})
        assert p.beta_latent.item() == pytest.approx(np.arctanh(2 * 0.25 - 1))

    def test_properties_return_tensors(self):
        p = EpiParams(1000)
        assert torch.is_tensor(p.beta) and p.beta.ndim == 0

    def test_get_params_dict_returns_python_floats(self):
        d = EpiParams(1000).get_params_dict()
        assert set(d) == {"beta", "gamma", "mu"}
        assert all(type(v) is float for v in d.values())


# --------------------------------------------------------------------------- TorchStandardScaler


class TestTorchStandardScaler:
    """The minimal standard scaler, including its in-place mutation of the input."""

    def test_fit_uses_population_std(self):
        s = TorchStandardScaler()
        s.fit(np.array([1.0, 2.0, 3.0]), "cpu")
        assert s.mean.tolist() == [2.0]
        assert s.std.item() == pytest.approx(np.sqrt(2.0 / 3.0))

    def test_fit_on_2d_data_is_column_wise(self):
        s = TorchStandardScaler()
        s.fit(np.array([[1.0, 10.0], [3.0, 30.0]]), "cpu")
        assert s.mean.tolist() == [[2.0, 20.0]]
        assert s.std.shape == (1, 2)
        assert s.std[0].tolist() == pytest.approx([1.0, 10.0])

    def test_transform_tensor_values(self):
        s = TorchStandardScaler()
        s.fit(np.array([1.0, 2.0, 3.0]), "cpu")
        out = s.transform(torch.tensor([1.0, 2.0, 3.0]))
        assert out.tolist() == pytest.approx([-1.2247449, 0.0, 1.2247449], abs=1e-5)

    def test_transform_mutates_the_tensor_in_place(self):
        # NOTE: current behavior — possible bug: transform() works in place (x -= mean; x /= std)
        # and returns the very same object, so the caller's tensor is overwritten
        s = TorchStandardScaler()
        s.fit(np.array([1.0, 2.0, 3.0]), "cpu")
        x = torch.tensor([1.0, 2.0, 3.0])
        original = x.clone()
        out = s.transform(x)
        assert out is x
        assert not torch.equal(x, original)

    def test_transform_numpy_is_in_place_too(self):
        s = TorchStandardScaler()
        s.fit(np.array([1.0, 2.0, 3.0]), "cpu")
        x = np.array([1.0, 2.0, 3.0])
        out = s.transform(x)
        assert out is x
        assert x == pytest.approx([-1.2247449, 0.0, 1.2247449], abs=1e-5)

    def test_transform_integer_numpy_fails(self):
        s = TorchStandardScaler()
        s.fit(np.array([1.0, 2.0, 3.0]), "cpu")
        with pytest.raises(Exception):  # numpy refuses float→int in-place casting
            s.transform(np.array([1, 2, 3]))

    def test_inverse_transform_round_trip(self):
        s = TorchStandardScaler()
        s.fit(np.array([5.0, 7.0, 9.0]), "cpu")
        x = torch.tensor([5.0, 7.0, 9.0])
        back = s.inverse_transform(s.transform(x))
        assert back is x
        assert back.tolist() == pytest.approx([5.0, 7.0, 9.0], abs=1e-5)

    def test_fit_transform(self):
        s = TorchStandardScaler()
        out = s.fit_transform(torch.tensor([2.0, 4.0]), "cpu")
        assert out.tolist() == pytest.approx([-1.0, 1.0], abs=1e-6)

    def test_constant_column_does_not_divide_by_zero(self):
        s = TorchStandardScaler()
        s.fit(np.array([3.0, 3.0, 3.0]), "cpu")
        out = s.transform(torch.tensor([3.0, 4.0]))
        assert out[0].item() == 0.0
        assert out[1].item() == pytest.approx(1.0 / 1e-7, rel=1e-3)


# --------------------------------------------------------------------------- EINN_PINN


@pytest.fixture
def pinn(pinn_arrays):
    t, S, I, R, D, population, train_size = pinn_arrays
    return EINN_PINN(t=t, S_data=S, I_data=I, R_data=R, D_data=D, population=population, train_size=train_size,
                     init_params={"beta": 0.1, "gamma": 0.06, "mu": 0.003})


class TestEinnPinnConstruction:
    """What EINN_PINN.__init__ builds: tensors, scalers fit on the train slice, network layout, optimizer parameter list."""

    def test_data_tensors_and_time_grad(self, pinn, pinn_arrays):
        t, S, I, R, D, population, train_size = pinn_arrays
        assert pinn.N == population and pinn.train_size == train_size and pinn.device == "cpu"
        assert pinn.t.requires_grad is True
        assert pinn.t.dtype == torch.float32
        assert torch.allclose(pinn.I_data, torch.tensor(I, dtype=torch.float32))
        assert pinn.S_data.requires_grad is False

    def test_scalers_are_fit_on_the_train_slice_only(self, pinn, pinn_arrays):
        _, S, I, _, _, _, train_size = pinn_arrays
        assert pinn.I_scaler.mean.item() == pytest.approx(I[:train_size].mean(), rel=1e-5)
        assert pinn.I_scaler.mean.item() != pytest.approx(I.mean(), rel=1e-3)
        assert pinn.S_scaler.mean.item() == pytest.approx(S[:train_size].mean(), rel=1e-5)

    def test_network_architecture(self, pinn):
        layers = list(pinn.state_net)
        assert [type(l) for l in layers] == [
            nn.Linear, nn.Tanh, nn.Dropout, nn.Linear, nn.Tanh, nn.Dropout, nn.Linear, nn.Tanh, nn.Dropout, nn.Linear
        ]
        assert [(l.in_features, l.out_features) for l in layers if isinstance(l, nn.Linear)] == [(1, 128), (128, 256), (256, 128), (128, 4)]
        assert all(l.p == 0.0 for l in layers if isinstance(l, nn.Dropout))

    def test_params_and_optimizer_list(self, pinn):
        assert isinstance(pinn.params, EpiParams)
        assert pinn.params.N == pinn.N
        assert pinn.params.beta.item() == pytest.approx(0.1, abs=1e-6)
        assert len(pinn.all_params) == 8 + 3  # 4 Linear layers × (weight, bias) + 3 latents
        assert sum(p.requires_grad for p in pinn.all_params) == 8

    def test_loss_histories_start_empty(self, pinn):
        assert pinn.losses == [] and pinn.losses_data == [] and pinn.losses_ode == []
        assert pinn.losses_ic == [] and pinn.losses_bc == []

    def test_default_init_params(self, pinn_arrays):
        t, S, I, R, D, population, train_size = pinn_arrays
        model = EINN_PINN(t, S, I, R, D, population, train_size)
        assert model.params.get_params_dict() == pytest.approx({"beta": 0.3, "gamma": 0.1, "mu": 0.04}, abs=1e-6)

    def test_network_starts_in_training_mode(self, pinn):
        assert pinn.state_net.training is True


class TestEinnPinnForward:
    """Forward pass, denormalization (softplus) and the autograd ODE residual."""

    def test_forward_shapes_and_non_negativity(self, pinn):
        t = torch.linspace(0, 29, 12).reshape(-1, 1)
        S, I, R, D = pinn.forward(t)
        for comp in (S, I, R, D):
            assert comp.shape == (12,)
            assert torch.all(comp >= 0)

    def test_denormalize_zero_is_softplus_of_the_mean(self, pinn):
        z = torch.zeros(3, 4)
        S, I, R, D = pinn.denormalize_states(z)
        assert torch.allclose(S, F.softplus(pinn.S_scaler.mean.expand(3)))
        assert torch.allclose(I, F.softplus(pinn.I_scaler.mean.expand(3)))

    def test_denormalize_mutates_its_input(self, pinn):
        # NOTE: current behavior — inverse_transform works in place on views of states_norm
        z = torch.zeros(3, 4)
        pinn.denormalize_states(z)
        assert not torch.equal(z, torch.zeros(3, 4))

    def test_ode_residual_shapes(self, pinn):
        t = pinn.t.reshape(-1, 1).detach().requires_grad_(True)
        out = pinn.compute_ode_residual(t)
        assert len(out) == 8
        for tensor in out:
            assert tensor.shape == (30,)
            assert torch.isfinite(tensor).all()

    def test_ode_residual_needs_a_differentiable_time_input(self, pinn):
        with pytest.raises(RuntimeError):
            pinn.compute_ode_residual(torch.zeros(5, 1))


class TestEinnPinnTraining:
    """train_model(): loss bookkeeping, weighted sum, frozen epidemic parameters, determinism."""

    def test_train_records_one_entry_per_epoch(self, pinn):
        pinn.train_model(n_epoch=3)
        for hist in (pinn.losses, pinn.losses_data, pinn.losses_ode, pinn.losses_ic, pinn.losses_bc):
            assert len(hist) == 3
            assert all(np.isfinite(hist))

    def test_total_loss_is_the_weighted_sum(self, pinn):
        pinn.train_model(n_epoch=2, lambda_data=2.0, lambda_ode=0.5, lambda_ic=0.25, lambda_bc=0.0)
        for i in range(2):
            expected = 2.0 * pinn.losses_data[i] + 0.5 * pinn.losses_ode[i] + 0.25 * pinn.losses_ic[i]
            assert pinn.losses[i] == pytest.approx(expected, rel=1e-5)

    def test_default_loss_weights(self, pinn):
        pinn.train_model(n_epoch=1)
        expected = pinn.losses_data[0] + 0.1 * pinn.losses_ode[0] + 0.1 * pinn.losses_ic[0] + 0.1 * pinn.losses_bc[0]
        assert pinn.losses[0] == pytest.approx(expected, rel=1e-5)

    def test_epidemic_parameters_never_move(self, pinn):
        before = pinn.params.get_params_dict()
        pinn.train_model(n_epoch=5)
        assert pinn.params.get_params_dict() == before

    def test_network_weights_do_move(self, pinn):
        first = pinn.state_net[0].weight.detach().clone()
        pinn.train_model(n_epoch=2)
        assert not torch.equal(pinn.state_net[0].weight, first)

    def test_progress_is_printed_every_thousand_epochs(self, pinn, capsys):
        pinn.train_model(n_epoch=2)
        out = capsys.readouterr().out
        assert "Epoch     0 | Loss:" in out
        assert "Params: β=0.1000, γ=0.0600, μ=0.0030" in out
        assert "Epoch     1" not in out

    def test_training_is_deterministic_under_a_fixed_seed(self, pinn_arrays):
        t, S, I, R, D, population, train_size = pinn_arrays
        runs = []
        for _ in range(2):
            torch.manual_seed(123)
            m = EINN_PINN(t, S, I, R, D, population, train_size)
            m.train_model(n_epoch=2)
            runs.append(m.losses)
        assert runs[0] == runs[1]


class TestEinnPinnPredict:
    """predict() and the MC-Dropout uncertainty estimate (predict_with_uncertainty)."""

    def test_predict_defaults_to_the_training_grid(self, pinn):
        S, I, R, D = pinn.predict()
        for comp in (S, I, R, D):
            assert comp.shape == (30,) and comp.device.type == "cpu" and comp.requires_grad is False

    def test_predict_on_a_custom_grid(self, pinn):
        S, I, R, D = pinn.predict(np.linspace(0, 60, 7))
        assert I.shape == (7,)

    def test_uncertainty_output_structure(self, pinn):
        t_values = np.linspace(0, 29, 10)
        res = pinn.predict_with_uncertainty(t_values=t_values, n_passes=6, dropout_rate=0.5)
        assert set(res) == {"mean", "std", "ci_lower_95", "ci_upper_95", "n_passes", "t"}
        assert res["n_passes"] == 6
        assert res["t"] is t_values
        for block in ("mean", "std", "ci_lower_95", "ci_upper_95"):
            assert set(res[block]) == {"S", "I", "R", "D"}
            assert res[block]["I"].shape == (10,)
        assert np.all(res["ci_lower_95"]["I"] <= res["ci_upper_95"]["I"])
        assert np.all(res["std"]["I"] >= 0)

    def test_dropout_makes_passes_differ(self, pinn):
        res = pinn.predict_with_uncertainty(n_passes=8, dropout_rate=0.5)
        assert res["std"]["I"].max() > 0

    def test_zero_dropout_gives_zero_spread(self, pinn):
        res = pinn.predict_with_uncertainty(n_passes=4, dropout_rate=0.0)
        assert np.all(res["std"]["I"] == 0)
        S, I, R, D = pinn.predict()
        assert np.allclose(res["mean"]["I"], I.numpy(), atol=1e-6)

    def test_dropout_probability_and_mode_are_restored(self, pinn):
        pinn.state_net.eval()
        pinn.predict_with_uncertainty(n_passes=2, dropout_rate=0.3)
        assert all(l.p == 0.0 for l in pinn.state_net if isinstance(l, nn.Dropout))
        assert pinn.state_net.training is False
        pinn.state_net.train()
        pinn.predict_with_uncertainty(n_passes=2, dropout_rate=0.3)
        assert pinn.state_net.training is True

    def test_latent_parameters_are_restored(self, pinn):
        before = pinn.params.get_params_dict()
        pinn.predict_with_uncertainty(n_passes=3, dropout_rate=0.3)
        assert pinn.params.get_params_dict() == before

    def test_uncertainty_defaults(self, pinn):
        res = pinn.predict_with_uncertainty()
        assert res["n_passes"] == 100
        assert res["mean"]["I"].shape == (30,)
