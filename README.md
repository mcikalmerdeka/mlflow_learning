# MLflow Tutorial

A comprehensive guide and hands-on examples for learning MLflow in data science MLOps workflows.

## Quick Setup

### 1. Install Dependencies with UV

```bash
# Install UV if you haven't already
pip install uv

# Install all dependencies
uv sync

# Or install individually
uv add mlflow scikit-learn pandas numpy matplotlib seaborn jupyter notebook ipykernel
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
```

### 3. Start the Tutorial

```bash
# Launch Jupyter Notebook
uv run jupyter notebook mlflow_examples.ipynb

# Or start MLflow UI (in a separate terminal)
uv run mlflow ui
```

## Files in This Repository

- **`MLflow_Guide.md`** - Comprehensive documentation and examples
- **`mlflow_examples.ipynb`** - Interactive Jupyter notebook with hands-on examples
- **`pyproject.toml`** - UV package configuration with all dependencies
- **`README.md`** - This quick setup guide

## Tutorial Contents

The tutorial covers:

1. **Basic MLflow Setup** - Installation and configuration with UV
2. **Experiment Tracking** - Logging parameters, metrics, and artifacts
3. **Model Comparison** - Comparing multiple models and algorithms
4. **Hyperparameter Tuning** - Using nested runs for parameter optimization
5. **Model Registry** - Registering, versioning, and staging models
6. **Advanced Features** - Autologging, custom artifacts, and visualizations
7. **Production Deployment** - Model serving and best practices

## Key MLflow Commands

```bash
# Start MLflow UI
uv run mlflow ui

# Serve a registered model
uv run mlflow models serve -m "models:/model_name/Production" -p 5000

# Run MLflow project
uv run mlflow run . -P param1=value1
```

## Next Steps

1. Follow the complete guide in `MLflow_Guide.md`
2. Run through the interactive examples in `mlflow_examples.ipynb`
3. Start the MLflow UI to explore your experiments
4. Adapt the examples to your own datasets and models

## Troubleshooting

- **Permission errors**: Check file permissions in the `mlruns` directory
- **Port conflicts**: Use different ports for MLflow UI (`mlflow ui --port 5001`)
- **Database locks**: Use separate tracking URIs for concurrent runs

For more detailed information, see the complete guide in `MLflow_Guide.md`. 