import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_workflow(data_path='data/sample_data.csv'):
    """Executa um fluxo de trabalho de ciência de dados de exemplo."""
    print(f"Carregando dados de: {data_path}")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo {data_path} não encontrado. Criando dados de exemplo.")
        # Criar dados de exemplo se o arquivo não existir
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        })
        data.to_csv(data_path, index=False)
        print(f"Dados de exemplo criados em: {data_path}")

    X = data[['feature1', 'feature2']]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Modelo treinado com sucesso. Acurácia: {accuracy:.2f}")
    return model, accuracy

if __name__ == "__main__":
    # Certificar-se de que o diretório de dados existe
    import os
    os.makedirs('data', exist_ok=True)
    run_workflow()

