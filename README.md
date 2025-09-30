# data-science-pro-workflow

[![CI/CD Pipeline](https://github.com/galafis/data-science-pro-workflow/actions/workflows/ci.yml/badge.svg)](https://github.com/galafis/data-science-pro-workflow/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/galafis/data-science-pro-workflow/branch/main/graph/badge.svg)](https://codecov.io/gh/galafis/data-science-pro-workflow)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Atribuição/Attribution: Todo o conteúdo deste repositório é de autoria de Gabriel Demetrios Lafis (@galafis).

---

PT-BR | English

# 🇧🇷 Visão Geral (Português)

Um repositório profissional completo que demonstra um workflow de Ciência e Análise de Dados de ponta a ponta, cobrindo: ingestão de dados, exploração, engenharia de features, modelagem (ML/DL), avaliação, versionamento de experimentos, MLOps (CI/CD), deploy, monitoramento e visualização de resultados. Projetado para o mercado, com exemplos práticos, boas práticas e automações.

Principais diferenciais:
- Estrutura modular e escalável (dados, notebooks, pipelines, modelos, APIs, dashboards)
- Padrões de boas práticas (PEP8, pre-commit, testes, tipagem, docstrings)
- Ferramentas modernas: Python, R, Docker, GitHub Actions, DVC/MLflow, FastAPI/Streamlit, Poetry, Makefile
- Dados sintéticos realistas + conectores para dados reais (APIs públicas)
- Conteúdo instrutivo com foco empregabilidade e portfólio

## Arquitetura do Projeto

```
.
├── data/                 # dados brutos, intermediários e processados (gitignored conforme política)
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/            # EDA, protótipos, experimentos (Jupyter, Rmarkdown)
│   ├── 01_eda_basica.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelagem_classificacao.ipynb
├── pipelines/            # DAGs e orquestração (Airflow/Prefect/Make)
│   ├── ingestao/
│   ├── transformacao/
│   └── treinamento/
├── src/                  # código fonte (pacote Python)
│   ├── dsworkflows/
│   │   ├── data/
│   │   │   ├── ingestion.py
│   │   │   ├── preprocess.py
│   │   │   └── validation.py
│   │   ├── features/
│   │   │   └── build_features.py
│   │   ├── models/
│   │   │   ├── train.py
│   │   │   ├── predict.py
│   │   │   └── evaluate.py
│   │   └── utils/
│   │       ├── io.py
│   │       └── metrics.py
│   └── __init__.py
├── dashboards/
│   ├── streamlit_app/
│   │   └── app.py
│   └── powerbi/          # modelos PBIX (referências e imagens)
├── api/
│   └── main.py           # FastAPI para servir modelos
├── ci/                   # workflows de CI/CD
│   └── github-actions/
│       └── ci.yml
├── tests/                # testes unitários e de integração
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── Makefile
├── pyproject.toml        # Poetry (dependências e build)
├── requirements.txt      # alternativa simples
├── dvc.yaml              # exemplo de pipeline DVC (opcional)
├── mlflow/               # pasta para artefatos/experimentos MLflow (se local)
├── LICENSE
└── README.md
```

## Como usar (rápido)

**Pré-requisitos:**
- Python 3.10+
- Docker (opcional, recomendado)
- Poetry (ou pip)

**Instalação com Poetry:**
```bash
poetry install
poetry shell
pre-commit install
```

**Ou com pip:**
```bash
pip install -r requirements.txt
```

**Rodar pipeline exemplo (Make + scripts):**
```bash
make data         # baixa/gera dados
make features     # engenharia de features
make train        # treina modelo
make evaluate     # avalia e salva métricas
make api          # inicia API FastAPI
make app          # inicia dashboard Streamlit
```

**Executar com Docker:**
```bash
docker compose up --build
```

## Tecnologias e Ferramentas

- **Linguagens**: Python, R (Rmd para relatórios)
- **Análise/ML**: pandas, numpy, scikit-learn, xgboost, lightgbm, pytorch/keras (exemplos)
- **Orquestração**: Makefile, Prefect/Airflow (sugestões de DAG)
- **Versionamento de dados/experimentos**: DVC, MLflow
- **APIs e Apps**: FastAPI, Streamlit
- **Visualização**: matplotlib, seaborn, plotly, Power BI (exemplos)
- **Qualidade**: pytest, pre-commit (black, isort, flake8, mypy), coverage
- **Infra**: Docker, docker-compose
- **CI/CD**: GitHub Actions (lint, testes, build, deploy)

## Dados de Exemplo

- **Synthetic**: geração controlada (sklearn.datasets, faker) para classificação e séries temporais
- **Reais (públicos)**:
  - OpenML (via API)
  - UCI ML Repository (links)
  - APIs públicas (ex: dados financeiros/mercado, clima)

Scripts e notebooks demonstram como baixar/limpar dados, garantir qualidade (Great Expectations/cerberus opcional) e versionar amostras.

## Exemplos Incluídos

- Notebook EDA com análise estatística, outliers e correlações
- Pipeline de classificação (fraude) com baseline, tuning e validação cruzada
- API para previsão em lote e em tempo real
- Dashboard interativo com KPIs e explicabilidade (SHAP)
- Exemplo de CI: lint + testes rodando em PRs

## Roadmap

- [ ] Adicionar exemplos com Timeseries (Prophet/Statsmodels)
- [ ] Treinamento distribuído (Spark/Ray) opcional
- [ ] Modelo de NLP (BERT) para sentimento multilíngue
- [ ] Monitoramento de drift e retraining automático

## Licença

Ver LICENSE. Cite "Gabriel Demetrios Lafis (@galafis)" ao reutilizar.

---

# 🇺🇸 Overview (English)

A comprehensive, production-minded Data Science and Analytics workflow repository that covers end-to-end steps: data ingestion, EDA, feature engineering, modeling (ML/DL), evaluation, experiment tracking, MLOps (CI/CD), deployment, monitoring, and visualization. Market-oriented, with practical examples, best practices, and automation.

**Key highlights:**
- Modular, scalable layout (data, notebooks, pipelines, models, APIs, dashboards)
- Professional standards (PEP8, pre-commit, tests, typing, docstrings)
- Modern stack: Python, R, Docker, GitHub Actions, DVC/MLflow, FastAPI/Streamlit, Poetry, Makefile
- Realistic synthetic data + connectors to real public datasets/APIs
- Instructional content targeting employability and portfolio impact

## Project Architecture

(See tree above.)

## Quickstart

**Requirements:** Python 3.10+, Docker (optional), Poetry or pip.

**Using Poetry:**
```bash
poetry install
poetry shell
pre-commit install
```

**Using pip:**
```bash
pip install -r requirements.txt
```

**Run sample pipeline:**
```bash
make data
make features
make train
make evaluate
make api
make app
```

**Docker:**
```bash
docker compose up --build
```

## Tech Stack

- **Languages**: Python, R
- **Analytics/ML**: pandas, numpy, scikit-learn, xgboost, lightgbm, pytorch/keras
- **Orchestration**: Makefile, Prefect/Airflow
- **Data/Experiment versioning**: DVC, MLflow
- **APIs & Apps**: FastAPI, Streamlit
- **Visualization**: matplotlib, seaborn, plotly, Power BI
- **Quality**: pytest, pre-commit (black, isort, flake8, mypy), coverage
- **Infra**: Docker, docker-compose
- **CI/CD**: GitHub Actions

## Sample Data

- Synthetic generators (sklearn.datasets, faker)
- Public sources: OpenML API, UCI ML Repo, finance/weather APIs

## Included Examples

- EDA notebook with stats, outliers, correlations
- Fraud-like classification pipeline: baseline → hyperparameter tuning → cross-validation
- Real-time and batch inference API
- Interactive dashboard with KPIs and explainability (SHAP)
- CI example for linting and tests on PRs

## Roadmap

- [ ] Timeseries module (Prophet/Statsmodels)
- [ ] Distributed training (Spark/Ray) optional
- [ ] Multilingual sentiment (BERT)
- [ ] Drift monitoring and automated retraining

## Author

All content authored by Gabriel Demetrios Lafis (@galafis).
