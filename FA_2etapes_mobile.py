# -*- coding: utf-8 -*-
"""
FA_2etapes_mobile.py  —  Version Streamlit (mobile-first)
Dimensionnement file d'attente 2 étapes — top 3 configs + heatmap

Lancement local :
    pip install streamlit simpy numpy pandas matplotlib openpyxl
    streamlit run FA_2etapes_mobile.py

Déploiement cloud gratuit :
    https://streamlit.io/cloud  (connecter le repo GitHub contenant ce fichier)
"""

from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import simpy
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────── Utils ───────────────────────────

def sec_to_h(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "--:--:--"
        x = float(x)
    except Exception:
        return "--:--:--"
    base = datetime(2025, 1, 1) + timedelta(seconds=x)
    return f"{base.hour:02d}:{base.minute:02d}:{base.second:02d}"

def safe_mean(seq):
    return float(np.mean(seq)) if len(seq) > 0 else np.nan

# ─────────────────────────── Moteur SimPy ───────────────────────────

def _simulate_once(cap1, cap2, *, rng, rate_l, rate_h,
                   dur1_l, dur1_h, dur2_l, dur2_h,
                   nb_max, t_sim):
    env = simpy.Environment()
    res1 = simpy.Resource(env, capacity=cap1)
    res2 = simpy.Resource(env, capacity=cap2)
    arrivals, exits = [], []

    def entity(i):
        arrivals.append((i, env.now))
        with res1.request() as r:
            yield r
            yield env.timeout(rng.uniform(dur1_l, dur1_h))
        with res2.request() as r:
            yield r
            yield env.timeout(rng.uniform(dur2_l, dur2_h))
            exits.append((i, env.now))

    def gen():
        i = 0
        while True:
            yield env.timeout(rng.uniform(rate_l, rate_h))
            i += 1
            env.process(entity(i))
            if i >= nb_max:
                break

    env.process(gen())
    env.run(until=t_sim)

    if not arrivals:
        return np.nan, 0
    arr  = dict(arrivals)
    exd  = dict(exits)
    ta   = safe_mean([exd[p] - arr[p] for p in arr if p in exd])
    enc  = len(arr) - len(exd)
    return ta, enc


def simulate_config(cap1, cap2, *, seed, n_reps,
                    rate_l, rate_h, dur1_l, dur1_h,
                    dur2_l, dur2_h, nb_max, t_sim):
    ta_list, ec_list = [], []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed + rep * 997 + cap1 * 10000 + cap2)
        ta, ec = _simulate_once(cap1, cap2, rng=rng,
                                rate_l=rate_l, rate_h=rate_h,
                                dur1_l=dur1_l, dur1_h=dur1_h,
                                dur2_l=dur2_l, dur2_h=dur2_h,
                                nb_max=nb_max, t_sim=t_sim)
        if not np.isnan(ta):
            ta_list.append(ta)
        ec_list.append(ec)
    return {
        "nb_e1":   cap1,
        "nb_e2":   cap2,
        "TA_moy":  float(np.mean(ta_list))  if ta_list  else np.nan,
        "TA_std":  float(np.std(ta_list))   if len(ta_list) > 1 else 0.0,
        "en_cours": int(round(np.mean(ec_list))) if ec_list else 0,
    }


def run_grid(params):
    results = []
    total = params["max_e1"] * params["max_e2"]
    bar   = st.progress(0, text="Simulation en cours…")
    done  = 0
    for i1 in range(1, params["max_e1"] + 1):
        for j1 in range(1, params["max_e2"] + 1):
            m = simulate_config(
                i1, j1,
                seed=params["seed"], n_reps=params["n_reps"],
                rate_l=params["rate_l"], rate_h=params["rate_h"],
                dur1_l=params["dur1_l"], dur1_h=params["dur1_h"],
                dur2_l=params["dur2_l"], dur2_h=params["dur2_h"],
                nb_max=params["nb_max"], t_sim=params["t_sim"],
            )
            nb_et = i1 + j1
            TA_c  = 1e9 if np.isnan(m["TA_moy"]) else m["TA_moy"]
            cost  = (i1 * params["c1"] + j1 * params["c2"]
                     + m["en_cours"] * params["c_nt"]
                     + TA_c * params["c_ta"])
            m["nb_etapes"] = nb_et
            m["Score"]     = round(cost * np.log(nb_et), 2)
            results.append(m)
            done += 1
            bar.progress(done / total, text=f"Config {done}/{total}…")
    bar.empty()
    return pd.DataFrame(results)


# ─────────────────────────── Export Excel ───────────────────────────

def df_to_excel(df, n1, n2):
    buf = BytesIO()
    rename = {
        "nb_e1":    f"nb {n1}",
        "nb_e2":    f"nb {n2}",
        "TA_moy":   "TA moyen (s)",
        "TA_std":   "TA écart-type (s)",
        "en_cours": "Non traités",
        "nb_etapes":"nb Étapes",
        "Score":    "Score",
    }
    df.rename(columns=rename).to_excel(buf, index=False)
    return buf.getvalue()


# ─────────────────────────── Heatmap ───────────────────────────

def make_heatmap(df, n1, n2):
    pivot = df.pivot_table(index="nb_e1", columns="nb_e2",
                           values="TA_moy", aggfunc="mean")\
              .sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel(f"nb {n2}", fontsize=8)
    ax.set_ylabel(f"nb {n1}", fontsize=8)
    ax.set_title("TA moyen (s) — toutes configs", fontsize=9)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# ─────────────────────────── App Streamlit ───────────────────────────

def main():
    st.set_page_config(
        page_title="File d'attente — 2 étapes",
        page_icon="⏱️",
        layout="centered",
    )

    st.title("⏱️ Dimensionnement file d'attente")
    st.caption("2 étapes — grid search SimPy")

    # ── Noms des étapes
    with st.expander("🏷️ Noms des étapes", expanded=True):
        c1, c2 = st.columns(2)
        n1 = c1.text_input("Étape 1", value="Étape 1")
        n2 = c2.text_input("Étape 2", value="Étape 2")

    # ── Capacités & durées
    with st.expander("⚙️ Capacités & durées", expanded=True):
        c1c, c2c = st.columns(2)
        max_e1 = c1c.number_input(f"Capacité max {n1}", 1, 50, 10, step=1)
        max_e2 = c2c.number_input(f"Capacité max {n2}", 1, 50, 15, step=1)

        st.markdown(f"**Durée {n1} (s)**")
        ca, cb = st.columns(2)
        dur1_l = ca.number_input("min", 1, 9999, 300, key="d1l")
        dur1_h = cb.number_input("max", 1, 9999, 480, key="d1h")

        st.markdown(f"**Durée {n2} (s)**")
        ca2, cb2 = st.columns(2)
        dur2_l = ca2.number_input("min", 1, 9999, 1200, key="d2l")
        dur2_h = cb2.number_input("max", 1, 9999, 1500, key="d2h")

    # ── Cadence & simulation
    with st.expander("🚶 Cadence & simulation"):
        ca, cb = st.columns(2)
        rate_l = ca.number_input("Espacement arrivées min (s)", 1, 9999, 300, key="rl")
        rate_h = cb.number_input("Espacement arrivées max (s)", 1, 9999, 600, key="rh")
        cc, cd = st.columns(2)
        t_sim  = cc.number_input("Durée simulation (s)", 600, 86400, 8*3600, step=600)
        nb_max = cd.number_input("Nb max entités", 10, 5000, 250, step=10)
        ce, cf = st.columns(2)
        n_reps = ce.number_input("Réplications / config", 1, 30, 5, step=1)
        seed   = cf.number_input("Seed RNG", 0, 99999, 42, step=1)

    # ── Coûts
    with st.expander("💰 Paramètres de coût (Score)"):
        ca, cb = st.columns(2)
        c_e1 = ca.number_input(f"Coût {n1} / unité", 0, 999999, 600)
        c_e2 = cb.number_input(f"Coût {n2} / unité", 0, 999999, 2000)
        cc, cd = st.columns(2)
        c_nt = cc.number_input("Coût non traités / pers", 0, 999999, 900)
        c_ta = cd.number_input("Coût TA / seconde", 0, 999999, 350)

    # ── Bouton
    st.divider()
    nb_configs = int(max_e1) * int(max_e2)
    st.caption(f"Grid search : {nb_configs} configs × {n_reps} réplication(s) = "
               f"{nb_configs * int(n_reps)} simulations")

    if st.button("▶ Lancer la simulation", type="primary", use_container_width=True):

        if dur1_l > dur1_h or dur2_l > dur2_h or rate_l > rate_h:
            st.error("Vérifie les bornes min/max (min ≤ max).")
            st.stop()

        params = dict(
            max_e1=int(max_e1), max_e2=int(max_e2),
            dur1_l=dur1_l, dur1_h=dur1_h,
            dur2_l=dur2_l, dur2_h=dur2_h,
            rate_l=rate_l, rate_h=rate_h,
            t_sim=t_sim, nb_max=int(nb_max),
            n_reps=int(n_reps), seed=int(seed),
            c1=c_e1, c2=c_e2, c_nt=c_nt, c_ta=c_ta,
        )

        df = run_grid(params)
        df_valid = df[np.isfinite(df["TA_moy"])].copy()
        df_valid.sort_values("Score", inplace=True, ignore_index=True)

        # ── TOP 3
        st.success("✅ Simulation terminée")
        st.subheader("🏆 Top 3 configurations")

        top3 = df_valid.head(3).copy()
        top3["TA (h:m:s)"] = top3["TA_moy"].apply(sec_to_h)
        top3["TA std (s)"]  = top3["TA_std"].apply(lambda x: int(round(x)))

        medals = ["🥇", "🥈", "🥉"]
        for i, (_, row) in enumerate(top3.iterrows()):
            with st.container(border=True):
                st.markdown(f"### {medals[i]}  Score : **{row['Score']:,.0f}**")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric(f"nb {n1}", int(row["nb_e1"]))
                col_b.metric(f"nb {n2}", int(row["nb_e2"]))
                col_c.metric("Total postes", int(row["nb_etapes"]))
                col_d, col_e, col_f = st.columns(3)
                col_d.metric("TA moyen", row["TA (h:m:s)"])
                col_e.metric("TA écart-type", f"{int(row['TA std (s)'])} s")
                col_f.metric("Non traités", int(row["en_cours"]))

        # ── Heatmap
        st.subheader("🗺️ Heatmap TA moyen")
        fig = make_heatmap(df_valid, n1, n2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Exports
        st.subheader("📥 Export")
        c_exp1, c_exp2 = st.columns(2)
        c_exp1.download_button(
            "💾 Top 3 (Excel)",
            data=df_to_excel(top3[["nb_e1","nb_e2","TA_moy","TA_std","en_cours","nb_etapes","Score"]], n1, n2),
            file_name="top3_configs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        c_exp2.download_button(
            "💾 Toutes configs (Excel)",
            data=df_to_excel(df_valid[["nb_e1","nb_e2","TA_moy","TA_std","en_cours","nb_etapes","Score"]], n1, n2),
            file_name="toutes_configs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
