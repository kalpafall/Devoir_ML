# Prédictions ML — application Streamlit

4 modèles, chacun avec une interface simple (formulaire) et une interface multiple (CSV) :

| Modèle | Type | Fichier(s) dans `models/` |
|---|---|---|
| Téléphone | Classification (`etat`) | `telephone_model.pkl` |
| Taxi | Régression (`Trip_Price`) | `rf_model.joblib`, `scaler.joblib`, `les_encodeurs.joblib` |
| Car data | Classification (`Venant_Occasion`) | `car_data_model.pkl` |
| Car dataset | Régression (`Selling_Price`) | `car_dataset_model.pkl` |

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
