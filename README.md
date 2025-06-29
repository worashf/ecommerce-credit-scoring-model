
# E-commerce Credit Scoring Model

[![GitHub stars](https://img.shields.io/github/stars/worashf/ecommerce-credit-scoring-model)](https://github.com/worashf/ecommerce-credit-scoring-model/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/worashf/ecommerce-credit-scoring-model)](https://github.com/worashf/ecommerce-credit-scoring-model/issues)

A machine learning model for credit risk assessment using e-commerce behavioral data, developed as part of Bati Bank's buy-now-pay-later (BNPL) initiative.

## 📌 Project Description

This repository contains an end-to-end credit scoring system that:
- Creates risk proxies from transaction data (RFM patterns, fraud flags)
- Engineers features from raw e-commerce behavioral data
- Trains and validates multiple model types
- Generates Basel II-compliant credit scores
- Provides API endpoints for real-time scoring

## 🎯 Learning Outcomes

This project is designed to develop the following competencies:

### 🛠️ Technical Skills
| Skill Area               | Implementation Example                                                                 |
|--------------------------|---------------------------------------------------------------------------------------|
| **Advanced scikit-learn** | Custom scoring metrics, pipeline engineering, and model stacking                      |
| **Feature Engineering**  | RFM feature creation, WOE binning, temporal feature extraction                        |
| **Model Development**    | Comparative implementation of Logistic Regression vs. XGBoost/LightGBM                |
| **CI/CD for ML**         | GitHub Actions workflows for model retesting and redeployment                         |
| **Python Logging**       | Structured logging for model training and inference pipelines                         |
| **Unit Testing**         | Pytest suites for data validation and model output verification                       |
| **Model Management**     | MLFlow integration for experiment tracking and model versioning                       |
| **MLOps**               | CML (Continuous Machine Learning) implementation for automated model evaluation       |

### 📚 Knowledge Development
| Knowledge Area           | Project Application                                                                   |
|--------------------------|---------------------------------------------------------------------------------------|
| **Business Context**      | Basel II compliance analysis and risk-return tradeoff evaluation                      |
| **Data Exploration**      | EDA of transaction patterns and fraud correlations                                    |
| **Predictive Analysis**  | Default probability estimation using behavioral proxies                               |
| **Machine Learning**     | End-to-end model development from EDA to production deployment                        |
| **Hyperparameter Tuning**| Bayesian optimization for credit model performance                                    |
| **Model Selection**      | Trade-off analysis between interpretability and performance in financial context      |

### 📝 Communication Skills
| Skill                    | Demonstration Artifact                                                                |
|--------------------------|---------------------------------------------------------------------------------------|
| **Technical Reporting**  | Jupyter notebooks with clear narrative explaining statistical methods                 |
| **Stakeholder Alignment**| Model documentation compliant with regulatory requirements                            |
| **Visual Communication** | SHAP value visualizations and score distribution plots                                |
| **Risk Communication**   | Documentation of model limitations and proxy variable uncertainty                     |

## 🛠️ Prerequisites

Before you begin, ensure you have:
- Python 3.10+
- Conda/Miniconda installed
- Git installed
- At least 8GB RAM (for model training)

## 🚀 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone git@github.com:worashf/ecommerce-credit-scoring-model.git
   cd ecommerce-credit-scoring-model
   ```

2. **Create and activate conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate credit-scoring-model
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Jupyter Lab**:
   ```bash
   jupyter lab
   ```

## 📂 Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml   # CI/CD pipeline
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Raw Xente dataset
│   └── processed/             # Processed features
├── notebooks/
│   └── 1.0-eda.ipynb          # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Feature engineering
│   ├── train.py               # Model training
│   ├── predict.py             # Inference
│   └── api/
│       ├── main.py            # FastAPI app
│       └── pydantic_models.py # API schemas
├── tests/
│   └── test_data_processing.py
├── environment.yml            # Conda environment
├── requirements.txt           # Pip requirements
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🌟 environment.yml

```yaml
name: credit-scoring-model
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyterlab
  - notebook
  - xgboost
  - lightgbm
  - ipywidgets
  - pip
  - pip:
      - shap
      - category_encoders
      - imbalanced-learn
      - mlflow
```

## 🎯 Credit Scoring Business Understanding

### Basel II Compliance
The Basel II Accord requires:
- Transparent risk weight calculations
- Documented model validation
- Clear differentiation of risk levels
Our model uses WOE-transformed features in logistic regression to satisfy these requirements while maintaining performance.

1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord places strong emphasis on accurate, transparent, and explainable risk measurement. Specifically, financial institutions must demonstrate that their credit risk models:

Are well understood by internal risk teams and regulators,

Have clear documentation of assumptions, data, and methodologies,

Support the calculation of regulatory capital in a way that’s transparent and justifiable.

In this context, the model we build must balance predictive performance with interpretability. Financial institutions are expected not just to automate credit decisions, but to explain and defend those decisions — especially in cases of credit rejection or audits. Hence, interpretability is not optional, but a regulatory necessity, especially when deploying internal rating-based (IRB) models for credit risk.

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
In the absence of a direct "default" label, we must define a proxy variable that approximates default risk — such as:

Fraudulent transaction behavior,

Patterns in RFM (Recency, Frequency, Monetary) data indicating non-repayment or drop-off in spending.

Creating a proxy is necessary to train a supervised model, but it introduces business risks:

Label leakage or bias: The proxy may unintentionally reflect biased signals (e.g., customers from certain regions or using certain channels being mislabeled).

Overfitting to non-generalizable behaviors: The proxy may correlate with non-financial signals that don’t hold in new environments.

Compliance risk: Using an unreliable proxy can lead to misclassification, which could mean denying credit to good customers or approving risky ones — damaging reputation, increasing losses, and potentially violating consumer protection laws.

Therefore, the definition and validation of the proxy variable must be done with care, business input, and fairness checks.


3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Aspect	Logistic Regression + WoE	Gradient Boosting (e.g., XGBoost)
Interpretability	High (good for audit, compliance)	Low (black-box by default)
Regulatory Acceptance	Preferred by regulators	Requires explainability tools (e.g., SHAP)
Performance	Moderate (linear boundaries)	High (non-linear, handles interactions)
Ease of Deployment	Easy, fast to compute scorecards	Needs robust infrastructure
Explainability Tools	Built-in (WoE, odds ratio)	Requires post-hoc tools (e.g., SHAP values)

In regulated contexts like credit scoring, simple models are often favored for compliance and explainability, even if they sacrifice some accuracy. Complex models can be used but must be interpreted, validated, and documented thoroughly, including fairness and robustness checks.



## 👨‍💻 Author

**Worash Abocherugn**  
📧: worashup@gmail.com  
🔗: [LinkedIn Profile](https://www.linkedin.com/in/worash-abocherugn/)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
