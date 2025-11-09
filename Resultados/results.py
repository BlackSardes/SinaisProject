from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=['Authentic (0)', 'Spoofed (1)']))

conf_matrix = confusion_matrix(y_test, y_pred)
print("\n--- Matriz de Confusão ---")
print(conf_matrix)

TN, FP, FN, TP = conf_matrix.ravel()

print(f"\nDetalhes do Desempenho (Baseado em Teste):")
print(f"Acurácia: {model.score(X_test, y_test):.4f}")
if (TP + FN) > 0:
    print(f"Sensibilidade (Detecção de Spoofing): {TP / (TP + FN):.4f}")
if (FP + TN) > 0:
    print(f"Falsa Detecção (Alarme Falso): {FP / (FP + TN):.4f}")

plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Authentic', 'Spoofed'])
plt.yticks(tick_marks, ['Authentic', 'Spoofed'])
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Predito')
plt.tight_layout()
plt.show()