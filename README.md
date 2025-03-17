🌺 Classificação de Flores com Algoritmos Baseados em Distância
Este projeto implementa diferentes classificadores baseados em distância para a classificação das flores do conjunto de dados Iris. O usuário pode escolher entre três métodos:
✅ Distância Mínima
✅ Distância Máxima
✅ Superfície de Decisão (Vizinho Mais Próximo)

🚀 Tecnologias Utilizadas
Python
Pandas (Manipulação de dados)
NumPy (Cálculos numéricos)
Matplotlib (Visualização de dados)
SciPy (Cálculo de distâncias)
Scikit-Learn (Divisão de dados em treino e teste)
📂 Instalação e Execução
🔹 1. Clonar o repositório
bash
Copiar
Editar
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
🔹 2. Criar e ativar um ambiente virtual (opcional, mas recomendado)
bash
Copiar
Editar
python -m venv venv  
source venv/bin/activate  # Linux/macOS  
venv\Scripts\activate  # Windows  
🔹 3. Instalar as dependências
bash
Copiar
Editar
pip install -r requirements.txt
🔹 4. Executar o script
bash
Copiar
Editar
python codigoatualizado.py
O programa solicitará:
1️⃣ Escolha do método de classificação (dist_min, dist_max ou superficie_decisao)
2️⃣ Entrada de 4 valores para previsão

Após isso, ele mostrará:
✅ A classe prevista para a entrada do usuário
✅ A acurácia do modelo
✅ Um gráfico de dispersão com a superfície de decisão

📊 Exemplo de Uso
Entrada no terminal:
lua
Copiar
Editar
Escolha o método (dist_min, dist_max, superficie_decisao): dist_min  
Digite os valores de x1, x2, x3, x4 separados por espaço: 5.1 3.5 1.4 0.2  
Saída esperada:
css
Copiar
Editar
A flor prevista para essa entrada é: Iris-setosa  
Acurácia: 95.67%  
(Gera um gráfico de dispersão com a separação das classes)  
