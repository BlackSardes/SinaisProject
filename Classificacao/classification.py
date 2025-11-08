## 3. CLASSIFICAÇÃO (ALGORITMOS DE ML)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Troquei para Random Forest, mais robusto com dados reais
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

if features_df_all.empty:
    print("ERRO: O DataFrame de características está vazio. Verifique os caminhos e a leitura dos arquivos.")
else:
    # 1. Preparação dos Dados
    # As características extraídas são os SQMs (Signal Quality Metrics)
    X = features_df_all[['C_N0', 'peak_to_secondary', 'fpw', 'asymmetry']].values
    y = features_df_all['label'].values
    
    # 2. Normalização/Escalonamento (Importante para SVM/NN, mas útil para RF também)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Divisão Treino/Teste
    # Garante que o mesmo arquivo não esteja no treino e teste se a aleatoriedade for importante
    # Mas aqui separamos por segmento, então a divisão aleatória é ok.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Dataset Total: {len(X)} amostras")
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    # 4. Treinamento do Modelo (Exemplo: Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Predição
    y_pred = model.predict(X_test)

    print("\nModelo treinado (Random Forest). Pronto para avaliação.")