"""
Application Streamlit : 4 modèles de Machine Learning
- Téléphone  : classification de l'état (D'occasion / Neuf / Reconditionné / Venant)
- Taxi       : régression du prix d'une course (Trip_Price)
- Car data   : classification Venant / D'occasion
- Car dataset: régression du prix de vente (Selling_Price)

Chaque modèle a deux interfaces :
- Simple   : un formulaire, une prédiction
- Multiple : upload d'un fichier CSV, une prédiction par ligne
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent / "models"


# --------------------------------------------------------------------------
# Description des 4 modèles
# --------------------------------------------------------------------------
# Pour chaque variable numérique : (valeur par défaut, minimum, pas)
MODELES = {
    "Téléphone (classification)": {
        "type": "classification",
        "description": "Prédire l'état d'un téléphone : D'occasion, Neuf, Reconditionné ou Venant.",
        "cible": "etat",
        "numeriques": {
            "prix": (250000, 0, 5000),
            "dim_ecr": (6.0, 0.0, 0.1),
            "ram": (8, 1, 1),
            "stockage": (128, 1, 1),
        },
        "labels": {
            "prix": "Prix (FCFA)",
            "adresse": "Adresse (quartier)",
            "marque": "Marque",
            "dim_ecr": "Dimension de l'écran (pouces)",
            "ram": "RAM (Go)",
            "stockage": "Stockage (Go)",
        },
    },
    "Taxi (régression)": {
        "type": "regression",
        "description": "Prédire le prix d'une course de taxi (Trip_Price, en $).",
        "unite": "$",
        "numeriques": {
            "Trip_Distance_km": (25.0, 0.0, 0.5),
            "Passenger_Count": (2, 1, 1),
            "Base_Fare": (3.5, 0.0, 0.1),
            "Per_Km_Rate": (1.2, 0.0, 0.05),
            "Per_Minute_Rate": (0.3, 0.0, 0.01),
            "Trip_Duration_Minutes": (60.0, 0.0, 1.0),
        },
        "labels": {
            "Trip_Distance_km": "Distance (km)",
            "Time_of_Day": "Moment de la journée",
            "Day_of_Week": "Type de jour",
            "Passenger_Count": "Nombre de passagers",
            "Traffic_Conditions": "Trafic",
            "Weather": "Météo",
            "Base_Fare": "Tarif de base",
            "Per_Km_Rate": "Tarif par km",
            "Per_Minute_Rate": "Tarif par minute",
            "Trip_Duration_Minutes": "Durée (minutes)",
        },
    },
    "Car data (classification)": {
        "type": "classification",
        "description": "Prédire si une voiture est « Venant » ou « D'occasion ».",
        "cible": "Venant_Occasion",
        "numeriques": {
            "Année": (2015, 1980, 1),
            "Prix": (7500000, 0, 100000),
        },
        "labels": {
            "Marque": "Marque",
            "Année": "Année",
            "Boite_Vitesse": "Boîte de vitesse",
            "Prix": "Prix (FCFA)",
            "Quartier": "Quartier",
        },
    },
    "Car dataset (régression)": {
        "type": "regression",
        "description": "Prédire le prix de vente d'une voiture (Selling_Price).",
        "unite": "",
        "numeriques": {
            "Kms_Driven": (70000, 0, 1000),
            "Present_Price": (15.0, 0.0, 0.5),
            "Age": (8, 0, 1),
        },
        "labels": {
            "Kms_Driven": "Kilométrage",
            "Present_Price": "Prix actuel (neuf)",
            "Fuel_Type": "Carburant",
            "Seller_Type": "Type de vendeur",
            "Transmission": "Transmission",
            "Age": "Âge (années)",
            "Selling_Price": "Prix de vente",
        },
    },
}

TAXI_FEATURES = [
    "Trip_Distance_km", "Time_of_Day", "Day_of_Week", "Passenger_Count",
    "Traffic_Conditions", "Weather", "Base_Fare", "Per_Km_Rate",
    "Per_Minute_Rate", "Trip_Duration_Minutes",
]
TAXI_CAT_COLS = ["Time_of_Day", "Day_of_Week", "Traffic_Conditions", "Weather"]


# --------------------------------------------------------------------------
# Chargement des modèles
# --------------------------------------------------------------------------
def charger_artefacts(nom):
    """Retourne un dictionnaire : model, scaler, encoders (dict col -> LabelEncoder), features."""
    if nom.startswith("Téléphone"):
        return joblib.load(MODELS_DIR / "telephone_model.pkl")
    if nom.startswith("Car data ("):
        return joblib.load(MODELS_DIR / "car_data_model.pkl")
    if nom.startswith("Car dataset"):
        return joblib.load(MODELS_DIR / "car_dataset_model.pkl")
    # Taxi : fichiers séparés, comme dans le notebook vu en classe
    encoders = joblib.load(MODELS_DIR / "les_encodeurs.joblib")  # liste dans l'ordre de TAXI_CAT_COLS
    return {
        "model": joblib.load(MODELS_DIR / "rf_model.joblib"),
        "scaler": joblib.load(MODELS_DIR / "scaler.joblib"),
        "encoders": dict(zip(TAXI_CAT_COLS, encoders)),
        "features": TAXI_FEATURES,
    }


def variables_categorielles(artefacts):
    """Variables explicatives catégorielles (la cible est exclue)."""
    return [f for f in artefacts["features"] if f in artefacts["encoders"]]


# --------------------------------------------------------------------------
# Prédiction (indépendante de l'interface, pour pouvoir la tester)
# --------------------------------------------------------------------------
def preparer_dataframe(df, nom, artefacts):
    """Vérifie les colonnes et les valeurs d'un DataFrame brut. Retourne (df, erreurs)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Car data : on accepte aussi la colonne Adresse au format "Quartier,Région"
    if nom.startswith("Car data (") and "Quartier" not in df.columns and "Adresse" in df.columns:
        df["Quartier"] = df["Adresse"].astype(str).str.split(",", n=1).str[0].str.strip()

    erreurs = []
    manquantes = [f for f in artefacts["features"] if f not in df.columns]
    if manquantes:
        erreurs.append(f"Colonnes manquantes : {', '.join(manquantes)}")
        return df, erreurs

    for col in variables_categorielles(artefacts):
        df[col] = df[col].astype(str).str.strip()
        connues = set(artefacts["encoders"][col].classes_)
        inconnues = sorted(set(df[col]) - connues)
        if inconnues:
            erreurs.append(f"Valeurs inconnues dans « {col} » : {', '.join(inconnues[:10])}")

    for col in artefacts["features"]:
        if col not in artefacts["encoders"]:
            valeurs = pd.to_numeric(df[col], errors="coerce")
            if valeurs.isna().any():
                lignes = (valeurs[valeurs.isna()].index + 1).tolist()[:10]
                erreurs.append(f"Valeurs non numériques dans « {col} » (lignes {lignes})")
            df[col] = valeurs
    return df, erreurs


def predire(df, nom, artefacts):
    """df : DataFrame brut (valeurs en clair). Retourne le DataFrame avec la/les colonne(s) de prédiction."""
    config = MODELES[nom]
    x = df[artefacts["features"]].copy()
    for col in variables_categorielles(artefacts):
        x[col] = artefacts["encoders"][col].transform(x[col])
    x = artefacts["scaler"].transform(x.values.astype(float))
    model = artefacts["model"]

    resultat = df.copy()
    if config["type"] == "classification":
        encodeur_cible = artefacts["encoders"][config["cible"]]
        y = model.predict(x).astype(int)
        resultat["Prédiction"] = encodeur_cible.inverse_transform(y)
        if hasattr(model, "predict_proba"):
            resultat["Confiance (%)"] = (model.predict_proba(x).max(axis=1) * 100).round(1)
    else:
        resultat["Prédiction"] = np.round(model.predict(x), 2)
    return resultat


def exemple_csv(nom, artefacts, n=5):
    """Petit CSV d'exemple avec les bonnes colonnes, pour le mode multiple."""
    config = MODELES[nom]
    rng = np.random.default_rng(0)
    data = {}
    for col in artefacts["features"]:
        if col in artefacts["encoders"]:
            data[col] = rng.choice(artefacts["encoders"][col].classes_, n)
        else:
            defaut = config["numeriques"][col][0]
            data[col] = [defaut] * n
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

html, body, .stApp, p, h1, h2, h3, h4, h5, h6, label, input, textarea, select,
button, li, td, th, div, span, .stMarkdown {
    font-family: 'Montserrat', sans-serif !important;
}
/* on garde la police des petites icônes de Streamlit (flèches, menus) */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {
    font-family: 'Material Symbols Rounded' !important;
}

:root {
    --vert: #00C9A7;
    --bleu: #4D8BF5;
    --nuit: #0F172A;
    --degrade: linear-gradient(120deg, #00C9A7 0%, #4D8BF5 100%);
}
.stApp { background: #F5F8FC; }

/* Barre latérale sombre */
section[data-testid="stSidebar"] { background: var(--nuit); }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] h1 {
    background: var(--degrade);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Bandeau de titre */
.bandeau {
    background: var(--degrade);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 10px 30px rgba(77, 139, 245, 0.25);
}
.bandeau h1 { color: white; margin: 0; font-size: 2.1rem; font-weight: 800; }
.bandeau p { margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.95; }
.bandeau .badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 10px;
    letter-spacing: 0.03em;
}

/* Formulaire en carte blanche */
div[data-testid="stForm"] {
    background: white;
    border: none;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
}

/* Boutons en dégradé */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button {
    background: var(--degrade);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    transition: transform 0.1s ease, box-shadow 0.2s ease;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(0, 201, 167, 0.35);
}

/* Carte de résultat */
.resultat {
    background: white;
    border-left: 8px solid var(--vert);
    border-radius: 16px;
    padding: 22px 28px;
    margin-top: 18px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
}
.resultat .titre { color: #64748B; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
.resultat .valeur {
    font-size: 2.4rem;
    font-weight: 800;
    background: var(--degrade);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.resultat .detail { color: #475569; font-size: 0.95rem; }
</style>
"""


def bandeau(nom):
    config = MODELES[nom]
    titre, _, type_modele = nom.partition(" (")
    st.markdown(
        f'''<div class="bandeau">
            <span class="badge">{type_modele.rstrip(")").upper()}</span>
            <h1>{titre}</h1>
            <p>{config["description"]}</p>
        </div>''',
        unsafe_allow_html=True,
    )


def carte_resultat(titre, valeur, detail=""):
    st.markdown(
        f'''<div class="resultat">
            <div class="titre">{titre}</div>
            <div class="valeur">{valeur}</div>
            <div class="detail">{detail}</div>
        </div>''',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Interfaces
# --------------------------------------------------------------------------
def interface_simple(nom, artefacts):
    config = MODELES[nom]
    labels = config["labels"]
    valeurs = {}
    with st.form("formulaire_simple"):
        colonnes = st.columns(2)
        for i, col in enumerate(artefacts["features"]):
            with colonnes[i % 2]:
                if col in artefacts["encoders"]:
                    options = list(artefacts["encoders"][col].classes_)
                    valeurs[col] = st.selectbox(labels.get(col, col), options)
                else:
                    defaut, minimum, pas = config["numeriques"][col]
                    valeurs[col] = st.number_input(labels.get(col, col), min_value=minimum,
                                                   value=defaut, step=pas)
        envoyer = st.form_submit_button("Prédire", type="primary", width="stretch")

    if envoyer:
        resultat = predire(pd.DataFrame([valeurs]), nom, artefacts).iloc[0]
        if config["type"] == "classification":
            detail = ""
            if "Confiance (%)" in resultat:
                detail = f"Confiance du modèle : {resultat['Confiance (%)']} %"
            carte_resultat("État prédit", resultat["Prédiction"], detail)
        else:
            valeur = f"{resultat['Prédiction']:,.2f} {config['unite']}".replace(",", " ").strip()
            carte_resultat("Prix prédit", valeur)


def interface_multiple(nom, artefacts):
    st.write("Chargez un fichier CSV avec une ligne par élément à prédire. "
             "Colonnes attendues : " + ", ".join(f"`{f}`" for f in artefacts["features"]))
    if nom.startswith("Car data ("):
        st.caption("Vous pouvez aussi fournir une colonne `Adresse` au format `Quartier,Région` "
                   "à la place de `Quartier`.")

    st.download_button("Télécharger un CSV d'exemple",
                       exemple_csv(nom, artefacts).to_csv(index=False).encode("utf-8"),
                       "exemple.csv", "text/csv")

    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is None:
        return
    try:
        df = pd.read_csv(fichier)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    df, erreurs = preparer_dataframe(df, nom, artefacts)
    if erreurs:
        for e in erreurs:
            st.error(e)
        return

    resultat = predire(df, nom, artefacts)
    st.success(f"{len(resultat)} prédictions effectuées.")
    st.dataframe(resultat, width="stretch")
    if MODELES[nom]["type"] == "classification":
        st.bar_chart(resultat["Prédiction"].value_counts(), color="#4D8BF5")
    st.download_button("Télécharger les résultats",
                       resultat.to_csv(index=False).encode("utf-8"),
                       "predictions.csv", "text/csv")


@st.cache_resource
def charger_en_cache(nom):
    return charger_artefacts(nom)


def main():
    st.set_page_config(page_title="Prédictions ML", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.sidebar.title("Prédictions ML")
    nom = st.sidebar.selectbox("Modèle", list(MODELES))
    mode = st.sidebar.radio("Interface", ["Simple", "Multiple"])

    bandeau(nom)

    try:
        artefacts = charger_en_cache(nom)
    except FileNotFoundError as e:
        st.error(f"Fichier du modèle introuvable : {e.filename}. "
                 "Vérifiez le dossier `models/`.")
        return

    if mode == "Simple":
        interface_simple(nom, artefacts)
    else:
        interface_multiple(nom, artefacts)


if __name__ == "__main__":
    main()
