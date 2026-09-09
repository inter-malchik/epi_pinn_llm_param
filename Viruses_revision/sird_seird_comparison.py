"""SIRD vs SEIRD on the Saint Petersburg COVID-19 series actually used in the paper.

Data: `covid-19_Kouprianov.csv` from the Epi_PINN_LLM_param repository. Its column I is
`ACTIVE.sk` of Kouprianov's SPb.COVID-19.united.csv starting on 2020-07-05 (369 days,
exact match), N = 6 000 000. Peak: day 199 = 2021-01-20. Training window of the paper:
first 185 days (pre-peak).

Two settings are compared:
  A. model selection on the full 369-day series (identifiable: the peak shape pins R0)
  B. the paper's forecasting protocol: fit on days 0-184, forecast days 185-368

Models: SIRD; SEIRD with (a) sigma and E0 free, (b) sigma = 1/5.2 d^-1 and E0 free,
(c) sigma free and E0 quasi-stationary (dE/dt = 0 at t = 0). In setting B an extra
"R/D-informed" SIRD is added: gamma and mu are moment estimates from the observed
recovered/death flows, only beta is fitted (k = 1).

AIC/BIC follow the paper's Eq. (aicbic): n ln(RSS/n) + 2k, n ln(RSS/n) + k ln n.
Outputs (next to this script): results.json, results_table.tex, figures/*.png|pdf
Run (from the repository root):  ./venv/bin/python Viruses_revision/sird_seird_comparison.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Phase 1 (model calibration)" / "real_datasets" / "covid-19_Kouprianov.csv"
DAY0 = "2020-07-05"
TRAIN = 185          # training window stated in the paper's text
TRAIN_ACTUAL = 120   # training window actually used for the real-data Phase 1 (SIRD_calibration.ipynb output, Fig. 3 split)
PAPER_SIRD = dict(beta=0.1219, gamma=0.0990, mu=0.0099)  # Phase-1 values reported in the paper
PAPER_PINN = dict(beta=0.0458, gamma=0.0371, mu=0.00229)  # PINN-recovered values reported in the paper
SIGMA_FIXED = 1 / 5.2  # mean incubation period 5.2 days (Li et al. 2020, NEJM)
SEED = 0
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------- data
df = pd.read_csv(DATA)
S_obs, I_obs, R_obs, D_obs = (df[c].to_numpy(float) for c in "SIRD")
S0, I0, R0, D0 = S_obs[0], I_obs[0], R_obs[0], D_obs[0]
N = S0 + I0 + R0 + D0
n_total = len(I_obs)
t_all = np.arange(n_total, dtype=float)
peak_day_obs, peak_val_obs = int(np.argmax(I_obs)), float(I_obs.max())
print(f"N={N:.0f}  I0={I0:.0f}  days={n_total}  train={TRAIN}  observed peak: day {peak_day_obs} ({peak_val_obs:.0f})")

# moment estimates of gamma and mu from the observed flows: dR/dt = gamma I, dD/dt = mu I
gamma_mom = (R_obs[TRAIN - 1] - R0) / np.sum(I_obs[: TRAIN - 1])
mu_mom = (D_obs[TRAIN - 1] - D0) / np.sum(I_obs[: TRAIN - 1])
print(f"moment estimates (train window): gamma={gamma_mom:.4f} mu={mu_mom:.5f}  (paper PINN: gamma={PAPER_PINN['gamma']}, mu={PAPER_PINN['mu']})")


# ----------------------------------------------------------------------------- models
def sird_rhs(y, t, beta, gamma, mu):
    S, I, R, D = y
    inf = beta * S * I / N
    return [-inf, inf - gamma * I - mu * I, gamma * I, mu * I]


def seird_rhs(y, t, beta, sigma, gamma, mu):
    S, E, I, R, D = y
    inf = beta * S * I / N
    return [-inf, inf - sigma * E, sigma * E - gamma * I - mu * I, gamma * I, mu * I]


def sim_sird(p, t):
    return odeint(sird_rhs, [S0, I0, R0, D0], t, args=tuple(p))[:, 1]


def sim_seird(p, t, E0):
    beta, sigma, gamma, mu = p
    return odeint(seird_rhs, [S0 - E0, E0, I0, R0, D0], t, args=(beta, sigma, gamma, mu))[:, 2]


def rss(pred, obs):
    return float(np.sum((pred - obs) ** 2))


def fit(objective, bounds, n_starts=20):
    """Multi-start L-BFGS-B (the local method of the paper's Phase 1) + differential
    evolution as a global check; returns the best parameter vector and its RSS."""
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    best = None
    for _ in range(n_starts):
        r = minimize(objective, lo + (hi - lo) * rng.random(len(bounds)), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    de = differential_evolution(objective, bounds, seed=SEED, maxiter=600, tol=1e-12, polish=True)
    if de.fun < best.fun:
        best = de
    return np.array(best.x), float(best.fun)


SIRD_BOUNDS = [(0.01, 0.99), (0.001, 0.5), (0.0001, 0.1)]  # as in SIRD_calibration.ipynb
SEIRD_BOUNDS = [(0.01, 0.99), (1 / 14, 1 / 2), (0.001, 0.5), (0.0001, 0.1)]
E0_BOUNDS = (0.0, 5 * I0)


def aic_bic(rss_, k, n):
    return n * np.log(rss_ / n) + 2 * k, n * np.log(rss_ / n) + k * np.log(n)


def fit_all_models(n_fit, extra_rd_informed=False):
    """Fit every model on days 0..n_fit-1; return rows (dict per model) and full-horizon curves."""
    t_fit, I_fit = t_all[:n_fit], I_obs[:n_fit]
    rows, curves = [], {}

    def add(name, k, params, rss_, curve, sigma=None, E0=None):
        beta, gamma, mu = params
        a, b = aic_bic(rss_, k, n_fit)
        rows.append(dict(model=name, k=k, beta=beta, sigma=sigma, gamma=gamma, mu=mu, E0=E0, R0=beta / (gamma + mu), rss=rss_, aic=a, bic=b))
        curves[name] = curve

    p, r = fit(lambda x: rss(sim_sird(x, t_fit), I_fit), SIRD_BOUNDS)
    add("SIRD", 3, p, r, sim_sird(p, t_all))

    if extra_rd_informed:
        p, r = fit(lambda x: rss(sim_sird((x[0], gamma_mom, mu_mom), t_fit), I_fit), [SIRD_BOUNDS[0]], n_starts=8)
        add("SIRD, γ/μ from R,D flows (β fitted)", 1, (p[0], gamma_mom, mu_mom), r, sim_sird((p[0], gamma_mom, mu_mom), t_all))

    x, r = fit(lambda x: rss(sim_seird(x[:4], t_fit, x[4]), I_fit), SEIRD_BOUNDS + [E0_BOUNDS])
    add("SEIRD (a) σ free, E0 free", 5, (x[0], x[2], x[3]), r, sim_seird(x[:4], t_all, x[4]), sigma=x[1], E0=x[4])

    x, r = fit(lambda x: rss(sim_seird((x[0], SIGMA_FIXED, x[1], x[2]), t_fit, x[3]), I_fit), [SEIRD_BOUNDS[0], SEIRD_BOUNDS[2], SEIRD_BOUNDS[3], E0_BOUNDS])
    add("SEIRD (b) σ=1/5.2, E0 free", 4, (x[0], x[1], x[2]), r, sim_seird((x[0], SIGMA_FIXED, x[1], x[2]), t_all, x[3]), sigma=SIGMA_FIXED, E0=x[3])

    x, r = fit(lambda x: rss(sim_seird(x, t_fit, x[0] * S0 * I0 / N / x[1]), I_fit), SEIRD_BOUNDS)
    E0c = x[0] * S0 * I0 / N / x[1]
    add("SEIRD (c) σ free, E0 quasi-stationary", 4, (x[0], x[2], x[3]), r, sim_seird(x, t_all, E0c), sigma=x[1], E0=E0c)

    for row in rows:
        row["delta_aic"] = row["aic"] - rows[0]["aic"]
        row["delta_bic"] = row["bic"] - rows[0]["bic"]
        pred = curves[row["model"]]
        pk = int(np.argmax(pred))
        row.update(peak_day_pred=pk, peak_day_error=pk - peak_day_obs, peak_height_pred=float(pred[pk]),
                   peak_height_rel_error=float((pred[pk] - peak_val_obs) / peak_val_obs))
        if n_fit < n_total:
            err = pred[n_fit:] - I_obs[n_fit:]
            row.update(rmse_test=float(np.sqrt(np.mean(err ** 2))), mae_test=float(np.mean(np.abs(err))),
                       r2_test=float(1 - np.sum(err ** 2) / np.sum((I_obs[n_fit:] - I_obs[n_fit:].mean()) ** 2)))
        else:
            err = pred - I_obs
            row.update(rmse_in=float(np.sqrt(np.mean(err ** 2))), mae_in=float(np.mean(np.abs(err))))
    return rows, curves


def show(title, rows):
    print(f"\n=== {title} ===")
    print("%-40s %2s %11s %8s %8s %7s %7s %6s %9s %9s %8s" % ("model", "k", "RSS", "AIC", "BIC", "dAIC", "dBIC", "R0", "RMSE", "MAE", "peak err"))
    for r in rows:
        rm, ma = (r.get("rmse_test"), r.get("mae_test")) if "rmse_test" in r else (r["rmse_in"], r["mae_in"])
        print("%-40s %2d %11.3e %8.1f %8.1f %7.1f %7.1f %6.2f %9.0f %9.0f %+8d" % (r["model"], r["k"], r["rss"], r["aic"], r["bic"], r["delta_aic"], r["delta_bic"], r["R0"], rm, ma, r["peak_day_error"]))
        print("      beta=%.4f gamma=%.4f mu=%.5f%s%s" % (r["beta"], r["gamma"], r["mu"], f" sigma={r['sigma']:.4f}" if r["sigma"] else "", f" E0={r['E0']:.0f}" if r["E0"] is not None else ""))


print("\n[Setting A] full series, n = %d" % n_total)
rows_A, curves_A = fit_all_models(n_total)
show("A: model selection on the full 369-day series", rows_A)
print("\n[Setting B] pre-peak training, n = %d, forecast on days %d-%d" % (TRAIN, TRAIN, n_total - 1))
rows_B, curves_B = fit_all_models(TRAIN, extra_rd_informed=True)
show("B: fit on days 0-184, forecast on 185-368", rows_B)

print("\n[Setting C] the paper's actual real-data protocol: training on %d days" % TRAIN_ACTUAL)
rows_C, curves_C = fit_all_models(TRAIN_ACTUAL, extra_rd_informed=True)
show("C: fit on days 0-119 (as in SIRD_calibration.ipynb / Fig. 3), forecast on 120-368", rows_C)

# the local optimum the notebook reported: L-BFGS-B from its own initial guess on the 120-day window
def _nb_obj(p):
    grid = np.linspace(0, 400, 1000)
    return np.mean((np.interp(t_all[:TRAIN_ACTUAL], grid, sim_sird(p, grid)) - I_obs[:TRAIN_ACTUAL]) ** 2)
gr = [np.log(I_obs[i] / I_obs[i - 1]) for i in range(1, 5)]
nb_init = [max(0.1, min(0.8, float(np.mean(gr)) + 0.1)), 0.1, 0.01]
nb_fit = minimize(_nb_obj, nb_init, method="L-BFGS-B", bounds=SIRD_BOUNDS, options={"maxiter": 1000})
notebook_reproduction = dict(init=nb_init, beta=float(nb_fit.x[0]), gamma=float(nb_fit.x[1]), mu=float(nb_fit.x[2]),
                             mse_train120=float(nb_fit.fun), R0=float(nb_fit.x[0] / (nb_fit.x[1] + nb_fit.x[2])))
print("\nnotebook reproduction (L-BFGS-B from init %s on 120 days): beta=%.4f gamma=%.4f mu=%.5f R0=%.3f MSE=%.0f" % (
    np.round(nb_init, 4).tolist(), *nb_fit.x, notebook_reproduction["R0"], nb_fit.fun))
print("  global SIRD optimum on the same 120 days: beta=%.4f gamma=%.4f mu=%.5f MSE=%.0f" % (
    rows_C[0]["beta"], rows_C[0]["gamma"], rows_C[0]["mu"], rows_C[0]["rss"] / TRAIN_ACTUAL))

# the paper's Phase-1 SIRD parameters evaluated on both settings
pred_paper = sim_sird((PAPER_SIRD["beta"], PAPER_SIRD["gamma"], PAPER_SIRD["mu"]), t_all)
paper_eval = dict(rss_full=rss(pred_paper, I_obs), rss_train=rss(pred_paper[:TRAIN], I_obs[:TRAIN]),
                  mse_train120=rss(pred_paper[:TRAIN_ACTUAL], I_obs[:TRAIN_ACTUAL]) / TRAIN_ACTUAL,
                  rmse_test_from120=float(np.sqrt(np.mean((pred_paper[TRAIN_ACTUAL:] - I_obs[TRAIN_ACTUAL:]) ** 2))),
                  peak_day=int(np.argmax(pred_paper)), peak_height=float(pred_paper.max()),
                  R0=PAPER_SIRD["beta"] / (PAPER_SIRD["gamma"] + PAPER_SIRD["mu"]))
print("\npaper's SIRD parameters: RSS(full)=%.3e RSS(train)=%.3e  peak day %d height %.0f  R0=%.3f" % (
    paper_eval["rss_full"], paper_eval["rss_train"], paper_eval["peak_day"], paper_eval["peak_height"], paper_eval["R0"]))
print("  vs. SIRD fitted on full series: RSS(full)=%.3e ; on train window: RSS(train)=%.3e" % (rows_A[0]["rss"], rows_B[0]["rss"]))

# gamma–mu profile for SIRD on the pre-peak window (identifiability illustration)
profile = []
for g in np.linspace(0.02, 0.2, 19):
    r = minimize(lambda x: rss(sim_sird((x[0], g, x[1]), t_all[:TRAIN]), I_obs[:TRAIN]), [0.1, 0.005], method="L-BFGS-B", bounds=[SIRD_BOUNDS[0], SIRD_BOUNDS[2]])
    profile.append(dict(gamma=float(g), beta=float(r.x[0]), mu=float(r.x[1]), R0=float(r.x[0] / (g + r.x[1])), rss=float(r.fun)))
print("\nγ profile on the pre-peak window: RSS range %.3e–%.3e (ratio %.2f), net growth β−γ−μ ≈ %.4f–%.4f" % (
    min(p["rss"] for p in profile), max(p["rss"] for p in profile), max(p["rss"] for p in profile) / min(p["rss"] for p in profile),
    min(p["beta"] - p["gamma"] - p["mu"] for p in profile), max(p["beta"] - p["gamma"] - p["mu"] for p in profile)))

# ----------------------------------------------------------------------------- outputs
out = dict(
    data=dict(file=str(DATA), source_column="ACTIVE.sk (Kouprianov, SPb.COVID-19.united.csv)", day0=DAY0, n_days=n_total,
              train_days=TRAIN, N=N, I0=I0, S0=S0, R0=R0, D0=D0, peak_day_obs=peak_day_obs, peak_value_obs=peak_val_obs,
              peak_date="2021-01-20", train_end_date="2021-01-05", last_date="2021-07-08"),
    moment_estimates=dict(gamma=gamma_mom, mu=mu_mom),
    paper_sird=PAPER_SIRD, paper_pinn=PAPER_PINN, paper_sird_evaluated=paper_eval,
    setting_A_full_series=rows_A, setting_B_prepeak185=rows_B, setting_C_prepeak120=rows_C,
    notebook_reproduction=notebook_reproduction, gamma_profile_prepeak=profile,
)
(HERE / "results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

LABELS = {
    "SIRD": "SIRD",
    "SIRD, γ/μ from R,D flows (β fitted)": r"SIRD, $\gamma,\mu$ from $R$/$D$ flows",
    "SEIRD (a) σ free, E0 free": r"SEIRD ($\sigma$, $E_0$ free)",
    "SEIRD (b) σ=1/5.2, E0 free": r"SEIRD ($1/\sigma = 5.2$~d, $E_0$ free)",
    "SEIRD (c) σ free, E0 quasi-stationary": r"SEIRD ($\sigma$ free, $E_0$ quasi-stationary)",
}


def tex_num(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def tex_table(rows, caption, label, test):
    head = r"\textbf{Model} & $k$ & \textbf{RSS} & \textbf{AIC} & \textbf{BIC} & $\Delta$\textbf{BIC} & $R_0$ & " + (
        r"\textbf{RMSE$_{\mathrm{test}}$} & \textbf{MAE$_{\mathrm{test}}$} & \textbf{Peak day error} \\" if test else r"\textbf{RMSE} & \textbf{Peak day error} \\")
    lines = [r"\begin{table}[H]", r"\caption{" + caption + "}", r"\centering", r"\begin{tabular}{l" + "c" * (9 if test else 8) + "}", r"\toprule", head, r"\midrule"]
    for r in rows:
        cells = [LABELS[r["model"]], str(r["k"]), tex_num(r["rss"]), f"{r['aic']:.1f}", f"{r['bic']:.1f}", f"{r['delta_bic']:+.1f}", f"{r['R0']:.2f}"]
        if test:
            cells += [f"{r['rmse_test']:,.0f}".replace(",", "{,}"), f"{r['mae_test']:,.0f}".replace(",", "{,}"), f"{r['peak_day_error']:+d}"]
        else:
            cells += [f"{r['rmse_in']:,.0f}".replace(",", "{,}"), f"{r['peak_day_error']:+d}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\label{" + label + "}", r"\end{table}"]
    return "\n".join(lines)


peak_tex = f"{peak_val_obs:,.0f}".replace(",", "{,}")
tex = tex_table(rows_A,
                r"Model selection on the full Saint Petersburg active-case series (369 days, 5 July 2020 -- 8 July 2021; least squares on $I(t)$). AIC/BIC per Equation~\eqref{eq:aicbic}, $\Delta$BIC relative to SIRD; RMSE over the whole series; observed peak on day 199 (20 January 2021), " + peak_tex + " active cases.",
                "tab:aicbic", test=False)
tex += "\n\n" + tex_table(rows_B,
                          r"Forecasting protocol of the paper: models calibrated on the pre-peak window (days 0--184, up to 5 January 2021) and evaluated on the held-out days 185--368. AIC/BIC computed on the training window ($n=185$).",
                          "tab:prepeak", test=True)
tex += "\n\n" + tex_table(rows_C,
                          r"The paper's actual real-data protocol: models calibrated on the first 120 days (5 July -- 1 November 2020) and evaluated on the held-out days 120--368. AIC/BIC computed on the training window ($n=120$).",
                          "tab:prepeak120", test=True)
(HERE / "results_table.tex").write_text(tex + "\n", encoding="utf-8")

# ----------------------------------------------------------------------------- figures
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"SIRD": ("#D32F2F", "-", 2.2), "SIRD, γ/μ from R,D flows (β fitted)": ("#6A1B9A", "-.", 2.0),
         "SEIRD (a) σ free, E0 free": ("#1565C0", "-", 2.0), "SEIRD (b) σ=1/5.2, E0 free": ("#2E7D32", "--", 1.8),
         "SEIRD (c) σ free, E0 quasi-stationary": ("#FF8C00", ":", 2.0)}

fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))
for ax, (title, rows, curves, split) in zip(axes, [
        ("A. Fit on the full series (model selection)", rows_A, curves_A, None),
        ("B. Fit on days 0–184 (as stated in the text), forecast of 185–368", rows_B, curves_B, TRAIN),
        ("C. Fit on days 0–119 (as actually done, Fig. 3), forecast of 120–368", rows_C, curves_C, TRAIN_ACTUAL)]):
    if split:
        ax.plot(t_all[:split], I_obs[:split], "o", ms=3, color="black", alpha=0.7, label="Observed, training")
        ax.plot(t_all[split:], I_obs[split:], "o", ms=3, mfc="white", mec="black", label="Observed, held out")
        ax.axvline(split, color="gray", ls="--", lw=1.2)
    else:
        ax.plot(t_all, I_obs, "o", ms=3, color="black", alpha=0.7, label="Observed")
    for name, pred in curves.items():
        c, ls, lw = STYLE[name]
        r = next(x for x in rows if x["model"] == name)
        extra = f", RMSE$_{{test}}$ {r['rmse_test']:,.0f}" if split else ""
        ax.plot(t_all, pred, ls, color=c, lw=lw, label=f"{name} (BIC {r['bic']:.0f}{extra})")
    if split == TRAIN_ACTUAL:
        ax.plot(t_all, pred_paper, "-", color="#795548", lw=2.2, label="Paper's Phase-1 SIRD (β=0.1219, local optimum)")
    ax.scatter([peak_day_obs], [peak_val_obs], s=90, marker="s", color="black", zorder=6, label="Observed peak (day 199)")
    ax.set_title(title); ax.set_xlabel("Day (day 0 = 5 July 2020)"); ax.set_ylabel("Active cases I(t)")
    ax.set_ylim(0, 2.6 * peak_val_obs); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Saint Petersburg, second COVID-19 wave: SIRD vs SEIRD", fontsize=13)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / "figures" / f"sird_seird_fit_forecast.{ext}", dpi=150)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot([p["gamma"] for p in profile], [p["rss"] for p in profile], "o-", color="#1565C0")
ax2.set_xlabel("γ (fixed)"); ax2.set_ylabel("RSS on days 0–184 (β, μ re-optimized)")
ax2.set_title("Pre-peak window: γ is not identifiable from I(t) alone")
ax2.grid(alpha=0.3); fig2.tight_layout()
fig2.savefig(HERE / "figures" / "sird_gamma_profile_prepeak.png", dpi=150)

# ----------------------------------------------------------------------------- results.pdf
# A self-contained PDF report (no LaTeX needed): the three tables of results_table.tex
# rendered with matplotlib, followed by the two figures.
from matplotlib.backends.backend_pdf import PdfPages

PDF_LABELS = {
    "SIRD": "SIRD",
    "SIRD, γ/μ from R,D flows (β fitted)": "SIRD, γ/μ from R/D flows (β fitted)",
    "SEIRD (a) σ free, E0 free": "SEIRD (a): σ, E₀ free",
    "SEIRD (b) σ=1/5.2, E0 free": "SEIRD (b): 1/σ = 5.2 d, E₀ free",
    "SEIRD (c) σ free, E0 quasi-stationary": "SEIRD (c): σ free, E₀ quasi-stationary",
}


def _sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m}·10^{int(e)}"


def table_page(pdf, title, subtitle, rows, test):
    page = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    page.text(0.05, 0.93, title, fontsize=14, weight="bold")
    page.text(0.05, 0.89, subtitle, fontsize=9, wrap=True)
    cols = ["Model", "k", "RSS", "AIC", "BIC", "ΔBIC", "R₀", "RMSE", "MAE", "Δ peak day", "β", "γ", "μ", "σ", "E₀"]
    widths = [0.20, 0.03, 0.07, 0.05, 0.05, 0.05, 0.045, 0.06, 0.06, 0.065, 0.05, 0.05, 0.055, 0.045, 0.045]
    cells = []
    for r in rows:
        rm, ma = (r["rmse_test"], r["mae_test"]) if test else (r["rmse_in"], r["mae_in"])
        cells.append([
            PDF_LABELS[r["model"]], str(r["k"]), _sci(r["rss"]), f"{r['aic']:.1f}", f"{r['bic']:.1f}", f"{r['delta_bic']:+.1f}",
            f"{r['R0']:.2f}", f"{rm:,.0f}", f"{ma:,.0f}", f"{r['peak_day_error']:+d}",
            f"{r['beta']:.4f}", f"{r['gamma']:.4f}", f"{r['mu']:.5f}",
            f"{r['sigma']:.3f}" if r["sigma"] else "–", f"{r['E0']:.0f}" if r["E0"] is not None else "–",
        ])
    ax = page.add_axes([0.03, 0.45, 0.94, 0.4])
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center", colWidths=widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.7)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#2E5496"); cell.set_text_props(color="white", weight="bold")
        elif ri % 2 == 0:
            cell.set_facecolor("#EEF2F7")
        if ci == 0:
            cell.set_text_props(ha="left"); cell.PAD = 0.02
    page.text(0.05, 0.60, "AIC = n·ln(RSS/n) + 2k,  BIC = n·ln(RSS/n) + k·ln n  (n = number of training points). ΔBIC relative to SIRD; negative favours the row.\n"
              + ("RMSE/MAE on the held-out days. " if test else "RMSE/MAE in-sample. ") + "Δ peak day = predicted − observed peak day (observed: day 199).",
              fontsize=8, color="#444444", va="top", linespacing=1.6)
    pdf.savefig(page); plt.close(page)


with PdfPages(HERE / "results.pdf") as pdf:
    # cover
    page = plt.figure(figsize=(11.69, 8.27))
    page.text(0.05, 0.85, "SIRD vs SEIRD on the Saint Petersburg second COVID-19 wave", fontsize=18, weight="bold")
    page.text(0.05, 0.78, "Results of sird_seird_comparison.py — companion to results.json / results_table.tex", fontsize=11, color="#444444")
    info = [
        f"Data: {DATA.name} (column I = ACTIVE.sk of Kouprianov's SPb.COVID-19.united.csv), day 0 = {DAY0}, {n_total} days, N = {N:,.0f}",
        f"Observed peak: day {peak_day_obs} (2021-01-20), {peak_val_obs:,.0f} active cases;  I₀ = {I0:.0f}, R(t=0) = {R0:.0f}, D(t=0) = {D0:.0f}",
        f"Moment estimates from the observed R/D flows over the first {TRAIN} days: γ = {gamma_mom:.4f}, μ = {mu_mom:.5f}",
        "",
        "Settings:",
        "  A — model selection: every model fitted to the full 369-day series (identifiable: the peak shape pins R₀)",
        "  B — protocol stated in the paper's text: fit on days 0–184, forecast on days 185–368",
        f"  C — protocol actually used for Phase 1 (SIRD_calibration.ipynb, Fig. 3): fit on days 0–{TRAIN_ACTUAL - 1}, forecast on days {TRAIN_ACTUAL}–368",
        "",
        "Fitting: least squares on I(t), multi-start L-BFGS-B (20 starts) + differential evolution, seed 0; bounds β∈[0.01,0.99], γ∈[0.001,0.5], μ∈[0.0001,0.1], σ∈[1/14,1/2], E₀∈[0,5·I₀].",
        "",
        f"Paper's Phase-1 SIRD parameters (β = {PAPER_SIRD['beta']}, γ = {PAPER_SIRD['gamma']}, μ = {PAPER_SIRD['mu']}, R₀ = {paper_eval['R0']:.3f}):",
        f"  MSE on the 120-day training window = {paper_eval['mse_train120']:,.0f} (global optimum: {rows_C[0]['rss'] / TRAIN_ACTUAL:,.0f});  RMSE on days {TRAIN_ACTUAL}–368 = {paper_eval['rmse_test_from120']:,.0f};",
        f"  predicted peak: day {paper_eval['peak_day']}, {paper_eval['peak_height']:,.0f} cases (observed: day {peak_day_obs}, {peak_val_obs:,.0f}).",
        f"  Reproduced as the L-BFGS-B local optimum from the notebook's initial guess {np.round(nb_init, 4).tolist()}: "
        f"β = {notebook_reproduction['beta']:.4f}, γ = {notebook_reproduction['gamma']:.4f}, μ = {notebook_reproduction['mu']:.5f}.",
    ]
    for i, line in enumerate(info):
        page.text(0.05, 0.70 - 0.035 * i, line, fontsize=9, family="DejaVu Sans")
    pdf.savefig(page); plt.close(page)

    table_page(pdf, "Table A — model selection on the full series (369 days)",
               "Least squares on I(t) over all 369 days. RMSE/MAE are in-sample. Observed peak: day 199.", rows_A, test=False)
    table_page(pdf, "Table B — protocol stated in the text: fit on days 0–184, forecast 185–368",
               "AIC/BIC on the training window (n = 185); RMSE/MAE and peak-day error on the held-out days 185–368.", rows_B, test=True)
    table_page(pdf, f"Table C — protocol actually used: fit on days 0–{TRAIN_ACTUAL - 1}, forecast {TRAIN_ACTUAL}–368",
               f"AIC/BIC on the training window (n = {TRAIN_ACTUAL}); RMSE/MAE and peak-day error on the held-out days {TRAIN_ACTUAL}–368. "
               f"The paper's parameters (β = 0.1219) give RMSE {paper_eval['rmse_test_from120']:,.0f} on the same held-out days.", rows_C, test=True)
    pdf.savefig(fig)   # the three-panel comparison figure
    pdf.savefig(fig2)  # γ profile
    pdf.infodict().update(Title="SIRD vs SEIRD — Saint Petersburg second wave", Author="sird_seird_comparison.py")

print("\nsaved results.json, results_table.tex, results.pdf, figures/")
