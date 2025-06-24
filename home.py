import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🏠 Hypotheek & Investering Optimalisatie met Risico en Multi-Scenario")

# --- Input sectie ---
st.sidebar.header("Instellingen")

woningprijs = st.sidebar.number_input("Woningprijs (€)", value=250000, step=5000)
totale_spaarpot = st.sidebar.number_input("Totale spaarpot (€)", value=150000, step=1000)
looptijden = st.sidebar.multiselect("Looptijden lening (jaren)", [15, 20, 25, 30], default=[25])
rentes = st.sidebar.multiselect("Rentevoeten (%)", [2.5, 3.0, 3.8, 4.5, 5.0], default=[3.8])
etf_rendement = st.sidebar.slider("Gemiddeld ETF-rendement (%)", 0.0, 15.0, 9.0) / 100
etf_volatiliteit = st.sidebar.slider("ETF Volatiliteit (standaarddeviatie %)", 0.0, 30.0, 15.0) / 100
simulaties = st.sidebar.number_input("Aantal Monte Carlo simulaties", min_value=1000, max_value=10000, value=3000, step=500)

# --- Functies ---
def maandlast(principal, annual_rate, years):
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return principal / n
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)

def totale_afbetaling(principal, annual_rate, years):
    return maandlast(principal, annual_rate, years) * years * 12

def monte_carlo_etf_return(startkapitaal, mean_return, volatility, years, n_simulaties=1000):
    # Simuleer jaarlijkse rendementen als normale verdeling, dan cumulatief product
    jaarlijkse_rendementen = np.random.normal(mean_return, volatility, size=(n_simulaties, years))
    groeiperioden = np.cumprod(1 + jaarlijkse_rendementen, axis=1)
    eindwaarden = startkapitaal * groeiperioden[:, -1]
    return eindwaarden

# --- Berekeningen & data verzamelen ---

results = []

# Loop over alle combinaties van looptijd en rente
for looptijd in looptijden:
    for rente in rentes:
        inbreng_values = np.arange(0, totale_spaarpot + 1, 1000)
        for E in inbreng_values:
            L = woningprijs - E
            if L < 0:
                continue
            maand = maandlast(L, rente/100, looptijd)
            totale_afbetaling_ = maand * looptijd * 12
            belegbaar = totale_spaarpot - E
            
            # Monte Carlo simulatie van ETF eindwaarden
            eindwaarden = monte_carlo_etf_return(belegbaar, etf_rendement, etf_volatiliteit, looptijd, simulaties)
            # Gemiddelde en 5e percentiel (worst case 95% confidence)
            mean_return_sim = np.mean(eindwaarden)
            p5_return_sim = np.percentile(eindwaarden, 5)
            
            netto_kost_mean = totale_afbetaling_ - mean_return_sim
            netto_kost_worst = totale_afbetaling_ - p5_return_sim
            
            results.append({
                "Looptijd (j)": looptijd,
                "Rente (%)": rente,
                "Inbreng (€)": E,
                "Lening (€)": L,
                "Maandlast (€)": maand,
                "Totale afbetaling (€)": totale_afbetaling_,
                "ETF Gem. opbrengst (€)": mean_return_sim,
                "ETF 5% worst case (€)": p5_return_sim,
                "Netto kost Gemiddeld (€)": netto_kost_mean,
                "Netto kost Worst case (€)": netto_kost_worst,
            })

df = pd.DataFrame(results)

# --- Optimalisaties per looptijd/rente ---
optimalisaties = []
grouped = df.groupby(["Looptijd (j)", "Rente (%)"])

for (looptijd, rente), group in grouped:
    min_kost_mean = group.loc[group["Netto kost Gemiddeld (€)"].idxmin()]
    min_kost_worst = group.loc[group["Netto kost Worst case (€)"].idxmin()]
    max_winst_mean = group.loc[group["Netto kost Gemiddeld (€)"].idxmax()]  # netto kost max = winst max negatief
    optimalisaties.append({
        "Looptijd (j)": looptijd,
        "Rente (%)": rente,
        "Min Netto Kost Gemiddeld": min_kost_mean["Netto kost Gemiddeld (€)"],
        "Inbreng Min Kost Gemiddeld": min_kost_mean["Inbreng (€)"],
        "Min Netto Kost Worst case": min_kost_worst["Netto kost Worst case (€)"],
        "Inbreng Min Kost Worst case": min_kost_worst["Inbreng (€)"],
    })

opt_df = pd.DataFrame(optimalisaties)

st.header("🧮 Samenvatting optimale inbreng per scenario")
st.dataframe(opt_df.style.format("{:,.2f}"))

# --- Interactieve filter & grafiek per scenario ---

st.header("📊 Analyse per looptijd en rente")

looptijd_sel = st.selectbox("Kies looptijd", looptijden)
rente_sel = st.selectbox("Kies rentevoet (%)", rentes)

df_sel = df[(df["Looptijd (j)"] == looptijd_sel) & (df["Rente (%)"] == rente_sel)]

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:red'
ax1.set_xlabel('Eigen Inbreng (€)')
ax1.set_ylabel('Netto Kost Gemiddeld (€)', color=color1)
ax1.plot(df_sel["Inbreng (€)"], df_sel["Netto kost Gemiddeld (€)"], color=color1, label="Netto Kost Gemiddeld")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axvline(df_sel.loc[df_sel["Netto kost Gemiddeld (€)"].idxmin(), "Inbreng (€)"], color=color1, linestyle='--')

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Netto Kost Worst case (€)', color=color2)
ax2.plot(df_sel["Inbreng (€)"], df_sel["Netto kost Worst case (€)"], color=color2, label="Netto Kost Worst case")
ax2.tick_params(axis='y', labelcolor=color2)
ax2.axvline(df_sel.loc[df_sel["Netto kost Worst case (€)"].idxmin(), "Inbreng (€)"], color=color2, linestyle='--')

plt.title(f"Netto Kosten bij Looptijd={looptijd_sel}j en Rente={rente_sel}%")
fig.tight_layout()
st.pyplot(fig)

# --- Detail weergave ---
st.header("📋 Details bij gekozen inbreng")

chosen_inbreng = st.slider("Selecteer eigen inbreng (€)", 0, int(totale_spaarpot), int(totale_spaarpot//2), step=1000)
details = df_sel[df_sel["Inbreng (€)"] == chosen_inbreng]

if not details.empty:
    row = details.iloc[0]
    st.markdown(f"""
    **Scenario details:**  
    - Looptijd: {row['Looptijd (j)']} jaar  
    - Rente: {row['Rente (%)']} %  
    - Eigen inbreng: €{row['Inbreng (€)']:,}  
    - Lening: €{row['Lening (€)']:,}  
    - Maandlast: €{row['Maandlast (€)']:.2f}  
    - Totale afbetaling lening: €{row['Totale afbetaling (€)']:,}  
    - Gemiddelde ETF opbrengst: €{row['ETF Gem. opbrengst (€)']:,}  
    - 5% Worst case ETF opbrengst: €{row['ETF 5% worst case (€)']:,}  
    - Netto kost (gemiddeld): €{row['Netto kost Gemiddeld (€)']:,}  
    - Netto kost (worst case): €{row['Netto kost Worst case (€)']:,}  
    """)

st.markdown("---")
st.markdown("Made by ChatGPT | Streamlit Hypotheek & ETF optimizer")

