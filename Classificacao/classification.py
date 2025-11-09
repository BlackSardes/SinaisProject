from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 

if 'features_df_all' not in locals() or features_df_all.empty:
    print("ERRO: O DataFrame de características está vazio. Execute o Módulo 2 primeiro para carregar features_df_all.")
else:
    X = features_df_all[['power_c_n0', 'sqm_peak_to_secondary', 'sqm_fpw', 'sqm_asymmetry']].values
    y = features_df_all['label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Dataset Total: {len(X)} amostras")
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModelo treinado (Random Forest). Pronto para avaliação.")
    