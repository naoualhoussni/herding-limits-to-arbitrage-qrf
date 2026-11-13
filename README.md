# 📈 Mimétisme Financier (Herding) & Limites à l'Arbitrage (LoA)
## Détection Non-Linéaire par Forêts Aléatoires Quantiles (QRF)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Target Journal](https://img.shields.io/badge/Target%20Journal-JBEF%20%2F%20Applied%20Econ%20Letters-brightgreen.svg)]()

> **Projet de Recherche & Data Science en Finance Empirique**  
> **Données :** Marchés A-Shares de Shanghai (SHA) & Shenzhen (SZA), base CSMAR (2003–2018, ~6,41 millions d'observations).  
> **Thématique 2 :** Détection Non-Linéaire du Mimétisme par Forêts Aléatoires Quantiles (QRF).

---

## 📌 Présentation du Projet

Cette recherche examine la dynamique non-linéaire entre **limites à l'arbitrage (Limits to Arbitrage — LoA)** et **mimétisme financier (herding behavior)** sur les marchés boursiers chinois.

Les régressions économétriques classiques par Moindres Carrés Ordinaires (MCO / Chang et al., 2000) à la moyenne affichent un très faible pouvoir explicatif ($R^2 < 5\%$) car **le mimétisme est intrinsèquement un phénomène de queue de distribution** qui surgit violemment lors des jours de stress et de panique de marché.

Pour combler cette lacune, ce projet déploie une approche conjointe :
1. **Étape 0 (Benchmark de Réplication) :** Reproduction intégrale de l'indice composite LoA à 6 facteurs et des modèles de référence (Chang et al., 2000; Christie & Huang, 1995) sur 3 886 jours de cotation.
2. **Régression Quantile Économétrique (Koenker & Bassett, 1978) :** Estimation de la courbure de mimétisme $\beta_2(\tau)$ sur la grille de déciles $\tau \in [0.05, 0.95]$.
3. **Quantile Regression Forests (QRF — Meinshausen, 2006) :** Modélisation non-paramétrique par arbres décisionnels quantiles avec validation croisée temporelle stricte (`TimeSeriesSplit`).
4. **Évaluation Hors-Échantillon :** Comparaison de la performance prédictive via la **Pinball Loss (Quantile Loss)** et le **RMSE**.

---

## 📊 Résultats Clés

### 1. Amplification du Mimétisme dans les Queues Inférieures
| Quantile ($\tau$) | Intercept $\beta_0$ | Linéaire $\|R_m\|$ $\beta_1$ | **Courbure Mimétisme $\beta_2$ ($R_m^2$)** | Lag CSAD $\beta_3$ |
| :---: | :---: | :---: | :---: | :---: |
| **$\tau = 0.05$ (Queue Extrême)** | 0.002164*** | 0.160299*** | **-2.806900*** (0.454) | 0.703965*** |
| $\tau = 0.10$ | 0.002481*** | 0.151897*** | **-2.651831*** (0.310) | 0.727828*** |
| $\tau = 0.20$ | 0.002935*** | 0.162367*** | **-2.701734*** (0.209) | 0.724575*** |
| $\tau = 0.50$ (Médiane) | 0.004128*** | 0.128690*** | **-1.777442*** (0.365) | 0.743549*** |
| $\tau = 0.95$ (Haut de distribution) | 0.008105*** | 0.090501*** | **-0.747126*** (0.336) | 0.799461*** |
| **MCO (Moyenne)** | 0.004259*** | 0.119308*** | **-1.191479*** (0.291) | 0.663258*** |

> **Constat majeur :** Dans le décile inférieur ($\tau = 0.05$), le mimétisme est **2,35 fois plus intense** ($\beta_2 = -2.81$) qu'à la moyenne estimée par MCO ($\beta_2 = -1.19$). Les régressions classiques sous-estiment massivement la vulnérabilité du marché.

### 2. Supériorité Prédictive Hors-Échantillon (Pinball Loss)
L'algorithme **Quantile Regression Forests (QRF)** surpasse la référence MCO paramétrique dans les queues de distribution, réduisant la perte Pinball jusqu'à **$+3.59\%$** à $\tau = 0.05$.

---

## 📁 Architecture du Répertoire

```text
├── .gitignore                  # Fichiers à ignorer par Git (données brutes lourdes .pkl)
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation complète du projet
│
├── notebooks/                  # Livrable 1 : Notebooks Jupyter
│   └── Thematique2_Herding_QRF_Complet.ipynb  # Notebook principal reproductible de bout en bout
│
├── manuscript/                 # Livrable 2 : Manuscrit Scientifique
│   ├── Manuscrit_Scientifique_Herding_QRF.md  # Article de 5 150 mots (format journal Scopus)
│   ├── figures/                # Figures haute résolution (300 DPI)
│   │   ├── fig1_timeseries_csad_loa.png
│   │   ├── fig2_quantile_beta2_trajectory.png
│   │   ├── fig3_pinball_loss_comparison.png
│   │   └── fig4_qrf_fan_chart.png
│   └── tables/                 # Tableaux récapitulatifs (statistiques, coefficients, Pinball loss)
│       ├── table1_descriptive_sha.csv
│       ├── table_quantile_regression_coefficients.csv
│       └── table_out_of_sample_evaluation.csv
│
├── src/                        # Code source modulaire
│   ├── data_loader.py          # Chargement et validation des données
│   ├── models.py               # Implémentation OLS, QuantReg et QRF
│   └── evaluate.py             # Calcul Pinball Loss, RMSE et TimeSeriesSplit
│
└── data/                       # Données de marché agrégées
    ├── README.md               # Lien vers les fichiers bruts Google Drive
    ├── daily_market_sha.csv    # Série quotidienne Shanghai (N = 3 886)
    └── daily_market_sza.csv    # Série quotidienne Shenzhen (N = 3 886)
```

---

## 🚀 Installation & Exécution Rapide

### 1. Cloner le repository
```bash
git clone https://github.com/<votre-utilisateur>/herding-limits-to-arbitrage-qrf.git
cd herding-limits-to-arbitrage-qrf
```

### 2. Créer un environnement virtuel et installer les dépendances
```bash
python -m venv venv
# Sur Windows :
venv\Scripts\activate
# Sur Linux/Mac :
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Lancer le Notebook Jupyter
```bash
jupyter notebook notebooks/Thematique2_Herding_QRF_Complet.ipynb
```

---

## 👥 Auteurs & Références Académiques
* **Auteurs :** Groupe de Recherche en Finance Empirique & Data Science
* **Références principales :**
  * Chang, E. C., Cheng, J. W., & Khorana, A. (2000). *An examination of herd behavior in equity markets*. Journal of Banking & Finance.
  * Gu, M., Kang, W., & Xu, B. (2018). *Limits of arbitrage and idiosyncratic volatility*. Journal of Banking & Finance.
  * Koenker, R., & Bassett, G. (1978). *Regression quantiles*. Econometrica.
  * Meinshausen, N. (2006). *Quantile regression forests*. Journal of Machine Learning Research.
