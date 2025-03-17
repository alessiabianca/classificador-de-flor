import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

# Carregar os dados
file_path = "Iris data.xlsx"
df = pd.read_excel(file_path)

# Selecionar todas as características
X = df.iloc[:, 0:4].values  # X1, X2, X3, X4
y = df.iloc[:, -1].values  # Rótulos das classes

# Converter rótulos para valores numéricos
class_labels = {label: idx for idx, label in enumerate(np.unique(y))}
y = np.array([class_labels[label] for label in y])

# Garantir 15 amostras para cada classe no conjunto de teste
X_train, X_test, y_train, y_test = [], [], [], []
for label in np.unique(y):
    X_label = X[y == label]
    y_label = y[y == label]
    X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(
        X_label, y_label, test_size=15, random_state=42, shuffle=True
    )
    X_train.extend(X_train_label)
    X_test.extend(X_test_label)
    y_train.extend(y_train_label)
    y_test.extend(y_test_label)

X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# Função do classificador de distância mínima
def min_distance_classifier(X_train, y_train, X_test):
    class_means = {label: X_train[y_train == label].mean(axis=0) for label in np.unique(y_train)}
    return np.array([min(class_means, key=lambda label: np.linalg.norm(x - class_means[label])) for x in X_test])

# Função do classificador de distância máxima (erro devido a separabilidade linear)
def max_distance_classifier(X_train, y_train, X_test):
    class_means = {label: X_train[y_train == label].mean(axis=0) for label in np.unique(y_train)}
    return np.array([max(class_means, key=lambda label: np.linalg.norm(x - class_means[label])) for x in X_test])

# Função da superfície de decisão baseada no vizinho mais próximo
def decision_surface_classifier(X_train, y_train, X_test):
    X_train = np.array(X_train)  # Garante que X_train seja um array NumPy
    y_train = np.array(y_train)  # Garante que y_train seja um array NumPy
    
    return np.array([y_train[np.argmin(cdist([x], X_train, metric='euclidean'))] for x in X_test])

# Perguntar ao usuário qual método usar
method = input("Escolha o método (dist_min, dist_max, superficie_decisao): ")
if method == "dist_min":
    classifier_function = min_distance_classifier
    title = "Classificador de Distância Mínima"
elif method == "dist_max":
    classifier_function = max_distance_classifier
    title = "Classificador de Distância Máxima"
elif method == "superficie_decisao":
    classifier_function = decision_surface_classifier
    title = "Superfície de Decisão"
else:
    print("Método inválido! Tente novamente executando o código e digitando uma opção válida.")
    exit()

# Entrada do usuário para classificação
try:
    user_input = list(map(float, input("Digite os valores de x1, x2, x3, x4 separados por espaço: ").split()))
    if len(user_input) != 4:
        raise ValueError("Por favor, insira exatamente 4 valores.")
    user_class = classifier_function(X_train, y_train, [user_input])[0]
    predicted_flower = [name for name, idx in class_labels.items() if idx == user_class][0]
    print(f"A flor prevista para essa entrada é: {predicted_flower}")
except ValueError as e:
    print(f"Erro na entrada: {e}")
    exit()

# Fazer predições e calcular acurácia
y_pred = classifier_function(X_train, y_train, X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Acurácia: {accuracy * 100:.2f}%")

# Criar gráfico de dispersão com superfície de decisão
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = classifier_function(X_train[:, :2], y_train, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
colors = ['black', 'yellow', 'red']  # Cores para cada classe
for idx, label in enumerate(np.unique(y_test)):
    plt.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1], label=f"{list(class_labels.keys())[idx]}", alpha=0.6, color=colors[idx])
    centroid = X_train[y_train == label].mean(axis=0)
    plt.scatter(centroid[0], centroid[1], marker='+', s=200, color=colors[idx], label=f"Centroide {list(class_labels.keys())[idx]}")

plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.title(title)
plt.show()