"""ABC test vs. bandits when shoppers' tastes change during the month.

Three shirts (yellow, orange, green) compete for one front-page slot for 28 days.
In the "drifting" scenario yellow is the clear winner in week 1, then tastes move
to orange and finally green. In the "stable" scenario each shirt keeps its
month-average click rate all month. The click rates are made up to illustrate
the trade-off; they are not from the Open Bandit Dataset.

Strategies:
  abc_test           show each shirt to 1/3 of visitors all month (the A/B/C test)
  eps_greedy         epsilon-greedy with sample averages (A2 / L08)
  thompson           Beta-Bernoulli Thompson sampling that remembers everything
  thompson_forget    Thompson sampling that discounts old data every day

At the end of the month each strategy recommends the shirt with the highest
estimated click rate, which is what would go on the front page next month.

Run:  python experiments/drift_ab_vs_bandit.py
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHIRTS = ["Yellow", "Orange", "Green"]
DAYS = 28
VISITORS_PER_DAY = 2000
RUNS = 1000
EPSILON = 0.1
FORGET_PER_DAY = 0.8  # keep 80% of yesterday's evidence, so memory is about 5 days

# True click rate for each shirt, one row per week.
WEEKLY_CTR = np.array([
    [0.050, 0.030, 0.025],
    [0.035, 0.040, 0.030],
    [0.025, 0.045, 0.035],
    [0.025, 0.040, 0.050],
])

OUT_DIR = Path(__file__).resolve().parent / "figures"

# Chart colors: the validated reference palette (dataviz skill). A literal
# yellow/orange/green set fails the color-vision checks, so each line is also
# labeled directly.
SHIRT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def daily_ctr(scenario):
    drifting = np.repeat(WEEKLY_CTR, DAYS // len(WEEKLY_CTR), axis=0)
    if scenario == "drifting":
        return drifting
    return np.tile(drifting.mean(axis=0), (DAYS, 1))


def split_uniform(rng, n, k):
    return rng.multinomial(n, [1 / k] * k)


def run_strategy(name, ctr, rng):
    """Simulate RUNS independent months. Returns per-run clicks, shows, estimates, picks."""
    k = len(SHIRTS)
    shows = np.zeros((RUNS, k))
    clicks = np.zeros((RUNS, k))
    disc_clicks = np.zeros((RUNS, k))
    disc_misses = np.zeros((RUNS, k))
    share = np.zeros((DAYS, k))

    for day in range(DAYS):
        if name == "abc_test":
            alloc = np.array([split_uniform(rng, VISITORS_PER_DAY, k) for _ in range(RUNS)])
        elif name == "eps_greedy":
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(shows > 0, clicks / shows, 2.0)  # untried shirts look best
            est = est + rng.random(est.shape) * 1e-9  # break ties at random
            greedy = est.argmax(axis=1)
            n_explore = rng.binomial(VISITORS_PER_DAY, EPSILON, size=RUNS)
            alloc = np.array([split_uniform(rng, n, k) for n in n_explore])
            alloc[np.arange(RUNS), greedy] += VISITORS_PER_DAY - n_explore
        else:
            if name == "thompson_forget":
                disc_clicks *= FORGET_PER_DAY
                disc_misses *= FORGET_PER_DAY
                a, b = 1 + disc_clicks, 1 + disc_misses
            else:
                a, b = 1 + clicks, 1 + (shows - clicks)
            draws = rng.beta(a[:, None, :], b[:, None, :], size=(RUNS, VISITORS_PER_DAY, k))
            choice = draws.argmax(axis=2)
            alloc = np.stack([(choice == i).sum(axis=1) for i in range(k)], axis=1)

        new_clicks = rng.binomial(alloc, ctr[day])
        shows += alloc
        clicks += new_clicks
        disc_clicks += new_clicks
        disc_misses += alloc - new_clicks
        share[day] = alloc.mean(axis=0) / VISITORS_PER_DAY

    if name == "thompson_forget":
        estimate = (1 + disc_clicks) / (2 + disc_clicks + disc_misses)
    elif name == "thompson":
        estimate = (1 + clicks) / (2 + shows)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            estimate = np.where(shows > 0, clicks / shows, 0.0)
    pick = (estimate + rng.random(estimate.shape) * 1e-12).argmax(axis=1)
    return clicks.sum(axis=1), estimate, pick, share


def summarize(scenario, seed=0):
    ctr = daily_ctr(scenario)
    month_avg = ctr.mean(axis=0)
    best_avg, best_now = int(month_avg.argmax()), int(ctr[-1].argmax())
    oracle_fixed = VISITORS_PER_DAY * ctr[:, best_avg].sum()
    oracle_daily = VISITORS_PER_DAY * ctr.max(axis=1).sum()
    rows, shares = [], {}
    for name in ["abc_test", "eps_greedy", "thompson", "thompson_forget"]:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()
        total, estimate, pick, share = run_strategy(name, ctr, rng)
        seconds = time.perf_counter() - start
        shares[name] = share
        rows.append({
            "strategy": name,
            "clicks": total.mean(),
            "clicks_ci": 1.96 * total.std(ddof=1) / np.sqrt(RUNS),
            "lost_vs_daily_oracle": oracle_daily - total.mean(),
            "pick": [(pick == i).mean() for i in range(len(SHIRTS))],
            "est_bias": estimate.mean(axis=0) - month_avg,
            "seconds": seconds,
        })
    return {"scenario": scenario, "ctr": ctr, "month_avg": month_avg, "best_avg": best_avg,
            "best_now": best_now, "oracle_fixed": oracle_fixed, "oracle_daily": oracle_daily,
            "rows": rows, "shares": shares}


def print_report(res):
    names = SHIRTS
    print(f"\n## Scenario: {res['scenario']}")
    print("Month-average click rate: " + ", ".join(f"{n} {v:.3%}" for n, v in zip(names, res["month_avg"])))
    print(f"Best shirt on average: {names[res['best_avg']]} | best in the last week: {names[res['best_now']]}")
    print(f"Clicks if you always showed {names[res['best_avg']]}: {res['oracle_fixed']:.0f} | "
          f"if you always showed each day's best: {res['oracle_daily']:.0f}")
    print("| Strategy | Clicks (mean ± 95% CI) | Lost vs. each day's best | "
          + " | ".join(f"Recommends {n}" for n in names)
          + " | " + " | ".join(f"{n} estimate bias (pts)" for n in names) + " | Seconds |")
    print("|" + "---|" * (4 + 2 * len(names)))
    for r in res["rows"]:
        print(f"| {r['strategy']} | {r['clicks']:.0f} ± {r['clicks_ci']:.0f} | {r['lost_vs_daily_oracle']:.0f} | "
              + " | ".join(f"{p:.0%}" for p in r["pick"]) + " | "
              + " | ".join(f"{b * 100:+.2f}" for b in r["est_bias"]) + f" | {r['seconds']:.1f} |")


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


LABELS = {"abc_test": "ABC test", "eps_greedy": "ε-greedy", "thompson": "Thompson sampling",
          "thompson_forget": "Thompson, forgets old data"}


def plot_traffic(res, path):
    days = np.arange(1, DAYS + 1)
    fig = plt.figure(figsize=(11, 6.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.55, wspace=0.18)
    ax = fig.add_subplot(grid[0, :])
    style_axis(ax)
    for i, shirt in enumerate(SHIRTS):
        ax.step(days, res["ctr"][:, i] * 100, where="mid", color=SHIRT_COLORS[i], lw=1.8, label=f"{shirt} shirt")
        ax.annotate(f"{shirt} shirt", (days[-1] + 0.3, res["ctr"][-1, i] * 100), color=INK_2, fontsize=8, va="center")
    ax.set_xlim(1, DAYS + 3)
    ax.set_xticks([1, 8, 15, 22, 28])
    ax.set_xlabel("Day", color=MUTED, fontsize=8)
    ax.set_ylabel("True click rate (%)", color=INK_2, fontsize=9)
    ax.set_title("What shoppers actually wanted each day", loc="left", color=INK, fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.6), frameon=False, fontsize=8, labelcolor=INK_2)
    for j, name in enumerate(res["shares"]):
        ax = fig.add_subplot(grid[1, j])
        style_axis(ax)
        share = res["shares"][name] * 100
        ax.stackplot(days, share.T, colors=SHIRT_COLORS, edgecolor=SURFACE, linewidth=1.2)
        ax.set_ylim(0, 100)
        ax.set_xlim(1, DAYS)
        ax.set_title(LABELS[name], loc="left", color=INK, fontsize=9)
        ax.set_xlabel("Day", color=MUTED, fontsize=8)
        if j == 0:
            ax.set_ylabel("Share of visitors shown (%)", color=INK_2, fontsize=8)
        else:
            ax.set_yticklabels([])
    fig.suptitle(f"Where each strategy sent traffic ({res['scenario']} tastes; mean of {RUNS} simulated months)",
                 x=0.06, ha="left", color=INK, fontsize=12)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_earn_learn(results, path):
    names = [r["strategy"] for r in results[0]["rows"]]
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 2.6 * len(results)), facecolor=SURFACE)
    for row, res in enumerate(results):
        clicks = [r["clicks"] for r in res["rows"]]
        ci = [r["clicks_ci"] for r in res["rows"]]
        right = [r["pick"][res["best_avg"]] * 100 for r in res["rows"]]
        ax = axes[row, 0]
        style_axis(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.barh(y, clicks, xerr=ci, color=INK_2, height=0.55, error_kw={"ecolor": MUTED, "lw": 1})
        for yi, c in zip(y, clicks):
            ax.annotate(f"{c:,.0f}", (c, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.axvline(res["oracle_daily"], color=AXIS, lw=1, ls="--")
        ax.annotate("each day's best", (res["oracle_daily"], y.max() + 0.45), fontsize=7, color=MUTED, ha="right")
        ax.set_yticks(y, [LABELS[n] for n in names], fontsize=8, color=INK_2)
        ax.set_xlim(0, res["oracle_daily"] * 1.12)
        ax.set_title(f"Earn: clicks during the month ({res['scenario']})", loc="left", color=INK, fontsize=10)
        ax = axes[row, 1]
        style_axis(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.barh(y, right, color=INK_2, height=0.55)
        for yi, v in zip(y, right):
            ax.annotate(f"{v:.0f}%", (v, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.set_yticks(y, [""] * len(y))
        ax.set_xlim(0, 112)
        ax.set_title(f"Learn: % of months it recommends {SHIRTS[res['best_avg']]}, the best on average",
                     loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    results = [summarize("drifting"), summarize("stable")]
    for res in results:
        print_report(res)
        plot_traffic(res, OUT_DIR / f"traffic_{res['scenario']}.png")
    plot_earn_learn(results, OUT_DIR / "earn_vs_learn.png")
    print(f"\nFigures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
