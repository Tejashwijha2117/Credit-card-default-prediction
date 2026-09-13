"""Credit policy simulator — interactive layer over the default-risk model.

Reads a pre-scored test set rather than loading the model, so it starts instantly
and needs no xgboost/sklearn at runtime. Regenerate the CSV from the notebook.

    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- palette -----------------------------------------------------------------
# Fixed categorical slots, assigned by role and never cycled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, MUTED, GRID = "#e34948", "#898781", "rgba(137,135,129,0.25)"

DATA = Path(__file__).parent / "data" / "processed" / "scored_test.csv"

st.set_page_config(page_title="Credit Policy Simulator", page_icon="💳",
                   layout="wide", initial_sidebar_state="expanded")


# --- data --------------------------------------------------------------------
@st.cache_data
def load():
    """Sort once by score so every cutoff becomes an O(1) cumulative lookup.

    Approving 'everyone below cutoff t' is a prefix of the sorted array, so the
    whole curve is two cumsums. Changing LGD or margin only rescales them, which
    is what keeps the sliders instant on 6,000 rows.
    """
    d = pd.read_csv(DATA).sort_values("p", ignore_index=True)
    bad_exp = np.concatenate([[0.0], np.cumsum(d.exposure * d.y)])
    good_exp = np.concatenate([[0.0], np.cumsum(d.exposure * (1 - d.y))])
    bads = np.concatenate([[0], np.cumsum(d.y)])
    return d, bad_exp, good_exp, bads


try:
    df, CUM_BAD_EXP, CUM_GOOD_EXP, CUM_BADS = load()
except FileNotFoundError:
    st.error(f"Scored data not found at `{DATA}`.\n\n"
             "Run the export cell at the end of the notebook to generate it.")
    st.stop()

P = df.p.values
N = len(df)
TOTAL_BADS = int(df.y.sum())
BASE_RATE = df.y.mean()


def outcome(k, lgd, margin):
    """Portfolio outcome when the k lowest-risk accounts are approved."""
    loss = CUM_BAD_EXP[k] * lgd
    revenue = CUM_GOOD_EXP[k] * margin
    bad_appr = CUM_BADS[k]
    return {
        "n_approved": k,
        "approval_rate": k / N,
        "bad_rate": bad_appr / k if k else 0.0,
        "capture_rate": 1 - bad_appr / TOTAL_BADS,
        "loss": loss,
        "revenue": revenue,
        "profit": revenue - loss,
    }


def money(v):
    s = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e6:
        return f"{s}NT${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{s}NT${a/1e3:.0f}k"
    return f"{s}NT${a:.0f}"


# --- sidebar -----------------------------------------------------------------
st.sidebar.title("Assumptions")
st.sidebar.caption(
    "The recommended cutoff is a function of these, not a property of the model. "
    "Move them and watch the optimum move."
)

# Integer percentages so the slider label reads "75%". Passing fractions with a
# "%.0f%%" format rounds 0.75 to "1%", which is wrong and looks broken.
lgd = st.sidebar.slider(
    "Loss given default", 45, 95, 75, 1, format="%d%%",
    help="Share of the outstanding balance lost when an account defaults. "
         "Unsecured card recovery is ~25%.") / 100
margin = st.sidebar.slider(
    "Net margin on carried balance", 6, 33, 18, 1, format="%d%%",
    help="Annual margin on a revolving balance, after funding and operating "
         "cost.") / 100

break_even = margin / (margin + lgd)

# Profit at every possible approval count, given the current economics.
ks = np.arange(N + 1)
profit_curve = CUM_GOOD_EXP * margin - CUM_BAD_EXP * lgd
k_opt = int(np.argmax(profit_curve))
cut_opt = float(P[k_opt]) if k_opt < N else 1.0

# The no-model benchmark: decline anyone already 1+ months late.
rule_mask = df.pay_1.values < 1
rule_n = int(rule_mask.sum())
rule_bads = int((rule_mask & (df.y.values == 1)).sum())
rule = {
    "approval_rate": rule_n / N,
    "bad_rate": rule_bads / rule_n,
    "capture_rate": 1 - rule_bads / TOTAL_BADS,
    "profit": (df.exposure.values * rule_mask * (df.y.values == 0)).sum() * margin
              - (df.exposure.values * rule_mask * (df.y.values == 1)).sum() * lgd,
}
k_matched = int(round(rule["approval_rate"] * N))

st.sidebar.divider()
st.sidebar.subheader("Cutoff")
mode = st.sidebar.radio(
    "Set the approve/decline threshold",
    ["Profit-maximising", "Match the rules-based baseline", "Manual"],
    label_visibility="collapsed",
)

if mode == "Profit-maximising":
    k = k_opt
elif mode == "Match the rules-based baseline":
    k = k_matched
else:
    cut = st.sidebar.slider("Approve when predicted risk is below",
                            0.02, 1.0, round(cut_opt, 3), 0.005)
    k = int(np.searchsorted(P, cut))

cur = outcome(k, lgd, margin)
cutoff = float(P[k]) if k < N else 1.0
approve_all = outcome(N, lgd, margin)
matched = outcome(k_matched, lgd, margin)

st.sidebar.divider()
st.sidebar.metric("Break-even  (margin ÷ (margin + LGD))", f"{break_even:.3f}")
st.sidebar.caption(
    f"Profit-maximising cutoff is **{cut_opt:.3f}**. It sits below break-even "
    "because exposure is U-shaped across the risk spectrum — see Risk bands."
)


# --- header ------------------------------------------------------------------
st.title("Credit Policy Simulator")
st.caption(
    f"{N:,} held-out credit card accounts · {BASE_RATE:.1%} default rate · "
    f"{money(df.exposure.sum())} total exposure · XGBoost, ROC-AUC 0.7808, "
    "calibrated to within 0.2% of the realised rate"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Approval rate", f"{cur['approval_rate']:.1%}",
          f"{cur['approval_rate']-1:.1%} vs approve-all")
c2.metric("Bad rate on approved book", f"{cur['bad_rate']:.2%}",
          f"{cur['bad_rate']-BASE_RATE:.2%}", delta_color="inverse")
c3.metric("Defaults declined", f"{cur['capture_rate']:.0%}",
          f"{int(TOTAL_BADS*cur['capture_rate']):,} of {TOTAL_BADS:,}")
c4.metric("Credit loss", money(cur["loss"]),
          money(cur["loss"] - approve_all["loss"]), delta_color="inverse")
c5.metric("Net profit", money(cur["profit"]),
          money(cur["profit"] - approve_all["profit"]))

if approve_all["profit"] < 0:
    st.info(
        f"**Approving every account loses {money(abs(approve_all['profit']))}.** "
        f"At a {BASE_RATE:.1%} default rate, a {margin:.0%} margin cannot carry a "
        f"{lgd:.0%} loss — so the portfolio is only viable with a cutoff.",
        icon="⚠️",
    )


# --- curves ------------------------------------------------------------------
def style(fig, ylab, xlab):
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        xaxis=dict(title=xlab, gridcolor=GRID, zeroline=False),
        yaxis=dict(title=ylab, gridcolor=GRID, zeroline=False),
    )
    return fig


# Thin the curve for plotting; 6,000 points is more than the eye resolves.
step = max(1, N // 400)
idx = np.arange(0, N + 1, step)
appr = idx / N
# Below ~300 approved the observed bad rate swings several points per account.
readable = idx >= 300
bad_rates = np.divide(CUM_BADS[idx], np.maximum(idx, 1))

left, right = st.columns(2)

with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=appr[readable], y=bad_rates[readable], mode="lines",
        line=dict(color=BLUE, width=2.5), name="Model cutoff",
        hovertemplate="approve %{x:.1%}<br>bad rate %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[rule["approval_rate"]], y=[rule["bad_rate"]], mode="markers",
        marker=dict(color=YELLOW, size=13, line=dict(color="white", width=2)),
        name="Rule: decline 1+ month late",
        hovertemplate="approve %{x:.1%}<br>bad rate %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[cur["approval_rate"]], y=[cur["bad_rate"]], mode="markers",
        marker=dict(color=ORANGE, size=15, line=dict(color="white", width=2)),
        name="Current policy",
        hovertemplate="approve %{x:.1%}<br>bad rate %{y:.2%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(style(fig, "Default rate among approved", "Approval rate"),
                    use_container_width=True)
    st.caption("**Approve more, and the book gets worse.** Every point is a "
               "candidate policy; the curve is the menu you actually choose from.")

with right:
    # Revenue, loss and profit are all currency, so they share one axis honestly.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=appr, y=CUM_GOOD_EXP[idx] * margin, mode="lines",
                             line=dict(color=AQUA, width=2), name="Revenue",
                             hovertemplate="%{y:,.0f}<extra>Revenue</extra>"))
    fig.add_trace(go.Scatter(x=appr, y=CUM_BAD_EXP[idx] * lgd, mode="lines",
                             line=dict(color=RED, width=2), name="Credit loss",
                             hovertemplate="%{y:,.0f}<extra>Loss</extra>"))
    fig.add_trace(go.Scatter(x=appr, y=profit_curve[idx], mode="lines",
                             line=dict(color=BLUE, width=3), name="Net profit",
                             hovertemplate="%{y:,.0f}<extra>Profit</extra>"))
    fig.add_trace(go.Scatter(
        x=[cur["approval_rate"]], y=[cur["profit"]], mode="markers",
        marker=dict(color=ORANGE, size=15, line=dict(color="white", width=2)),
        name="Current policy", hovertemplate="%{y:,.0f}<extra>Current</extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, f"NT$ over {N:,} accounts", "Approval rate"),
                    use_container_width=True)
    st.caption("**Profit is revenue minus loss, and the two cross.** The peak sits "
               "well short of approving everyone.")


# --- tabs --------------------------------------------------------------------
t1, t2, t3 = st.tabs(["Policy comparison", "Risk bands", "Sensitivity"])

with t1:
    st.subheader("Is the model better than the rule a bank runs without one?")
    st.markdown(
        "Beating *approve everyone* flatters any policy at all. The fair test is "
        "against **decline anyone already one or more months late** — a genuinely "
        "decent rule, since the model itself leans hard on recent repayment status "
        "— held to the **same approval rate**."
    )
    comp = pd.DataFrame([
        {"Policy": "Approve everyone", "Approval rate": approve_all["approval_rate"],
         "Bad rate": approve_all["bad_rate"], "Defaults declined": approve_all["capture_rate"],
         "Net profit": approve_all["profit"]},
        {"Policy": "Rule: decline 1+ month late", "Approval rate": rule["approval_rate"],
         "Bad rate": rule["bad_rate"], "Defaults declined": rule["capture_rate"],
         "Net profit": rule["profit"]},
        {"Policy": "Model, at the rule's approval rate", "Approval rate": matched["approval_rate"],
         "Bad rate": matched["bad_rate"], "Defaults declined": matched["capture_rate"],
         "Net profit": matched["profit"]},
        {"Policy": "Model, at its profit optimum", "Approval rate": outcome(k_opt, lgd, margin)["approval_rate"],
         "Bad rate": outcome(k_opt, lgd, margin)["bad_rate"],
         "Defaults declined": outcome(k_opt, lgd, margin)["capture_rate"],
         "Net profit": outcome(k_opt, lgd, margin)["profit"]},
    ])
    st.dataframe(
        comp.style.format({"Approval rate": "{:.1%}", "Bad rate": "{:.2%}",
                           "Defaults declined": "{:.0%}", "Net profit": "{:,.0f}"}),
        use_container_width=True, hide_index=True)

    lift = matched["profit"] - rule["profit"]
    st.success(
        f"**At a matched {rule['approval_rate']:.1%} approval rate, the model books a "
        f"{matched['bad_rate']:.1%} bad rate against the rule's {rule['bad_rate']:.1%}** "
        f"— worth {money(lift)} on {N:,} accounts. That difference is the model's "
        "actual contribution, and it is the number to quote.",
        icon="✅")

with t2:
    st.subheader("Where the risk — and the money — actually sits")
    b = df.copy()
    b["band"] = pd.qcut(b.p, 10, labels=False, duplicates="drop")
    g = b.groupby("band").agg(Accounts=("y", "size"), bad_rate=("y", "mean"),
                              mean_p=("p", "mean"),
                              mean_exposure=("exposure", "mean")).iloc[::-1]
    g.insert(0, "Decile", [f"D{i}" for i in range(10, 0, -1)])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=g.Decile, y=g.bad_rate, marker_color=BLUE,
                         name="Observed default rate",
                         hovertemplate="%{x}<br>%{y:.1%}<extra></extra>"))
    fig.add_hline(y=BASE_RATE, line=dict(color=MUTED, width=1),
                  annotation_text=f"portfolio {BASE_RATE:.1%}")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(style(fig, "Observed default rate", "Risk decile — D10 riskiest"),
                    use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=g.Decile, y=g.mean_exposure, marker_color=ORANGE,
                         name="Mean exposure",
                         hovertemplate="%{x}<br>NT$%{y:,.0f}<extra></extra>"))
    st.plotly_chart(style(fig, "Mean balance outstanding (NT$)",
                          "Risk decile — D10 riskiest"), use_container_width=True)

    st.warning(
        "**Exposure is U-shaped across the risk spectrum.** The safest accounts are "
        "large — transactors who put real money through the card and clear it. The "
        "middle deciles are small. Then the riskiest decile climbs back near the top: "
        "maxed-out revolvers carrying a balance they cannot clear.\n\n"
        "Declining a mid-risk account avoids a small loss; declining a top-decile "
        "account avoids a large one. **A policy tuned on account counts alone would "
        "systematically under-price exactly the segment doing the damage** — which is "
        "why the profit-maximising cutoff sits below the per-account break-even.",
        icon="📐")

    st.dataframe(
        g.style.format({"bad_rate": "{:.1%}", "mean_p": "{:.1%}",
                        "mean_exposure": "{:,.0f}", "Accounts": "{:,}"}),
        use_container_width=True, hide_index=True)

with t3:
    st.subheader("The cutoff moves with the economics, not the model")
    st.markdown(
        "Each cell is the profit-maximising cutoff for that pair of assumptions. "
        "Nothing about the model changes across this grid — only the business inputs."
    )
    margins = np.round(np.arange(0.06, 0.34, 0.03), 3)
    lgds = np.round(np.arange(0.45, 0.96, 0.05), 3)
    grid = np.zeros((len(lgds), len(margins)))
    for i, l in enumerate(lgds):
        for j, m in enumerate(margins):
            kk = int(np.argmax(CUM_GOOD_EXP * m - CUM_BAD_EXP * l))
            grid[i, j] = P[kk] if kk < N else 1.0

    fig = go.Figure(go.Heatmap(
        z=grid, x=[f"{m:.0%}" for m in margins], y=[f"{l:.0%}" for l in lgds],
        colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
        text=np.round(grid, 2), texttemplate="%{text}",
        textfont=dict(size=11), colorbar=dict(title="Cutoff"),
        hovertemplate="margin %{x}<br>LGD %{y}<br>cutoff %{z:.3f}<extra></extra>"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Net margin", yaxis_title="Loss given default")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"The optimal cutoff ranges **{grid.min():.2f} to {grid.max():.2f}** "
        f"({grid.max()/grid.min():.1f}×) across plausible assumptions. Two consequences:\n\n"
        "1. **A cutoff quoted without its assumptions is meaningless.** Anyone saying "
        "*we approve below 0.3* is implicitly asserting a margin and an LGD.\n"
        "2. **Re-tune the cutoff when the economics move, not when the model drifts.** "
        "A funding-cost change shifts the optimum without touching a single model weight "
        "— a different trigger, a different team, a different review cycle.",
        icon="🎚️")

st.divider()
st.caption(
    f"Current policy: approve when predicted default risk is below **{cutoff:.3f}** · "
    f"assumptions LGD {lgd:.0%}, margin {margin:.0%} · break-even {break_even:.3f}. "
    "Held-out backtest on one month of one portfolio — the curve is the deliverable, "
    "not its peak."
)