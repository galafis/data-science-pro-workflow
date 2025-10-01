import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.dsworkflows.workflow import run_workflow

# Mock para evitar a criação de arquivos CSV durante o teste
@pytest.fixture
def mock_data_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_data_path = data_dir / "sample_data.csv"
    pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }).to_csv(sample_data_path, index=False)
    return str(sample_data_path)

def test_run_workflow_returns_model_and_accuracy(mock_data_path):
    model, accuracy = run_workflow(data_path=mock_data_path)
    assert isinstance(model, LogisticRegression)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

def test_run_workflow_creates_data_if_not_found(tmp_path):
    # Simula um caminho onde o arquivo de dados não existe
    non_existent_path = tmp_path / "non_existent_data.csv"
    model, accuracy = run_workflow(data_path=str(non_existent_path))
    assert isinstance(model, LogisticRegression)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    assert non_existent_path.exists() # Verifica se o arquivo foi criado

