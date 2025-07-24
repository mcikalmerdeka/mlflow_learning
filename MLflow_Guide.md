# MLflow Guide: Complete Data Science MLOps Implementation

## Overview

MLflow is an open-source platform for managing the complete machine learning lifecycle. It tackles four primary functions:

1. **MLflow Tracking**: Record and query experiments (code, data, config, results)
2. **MLflow Projects**: Package ML code in reusable, reproducible form
3. **MLflow Models**: Deploy ML models in diverse serving environments
4. **MLflow Model Registry**: Centralized model store for managing model lifecycle

## Setup with UV

### Installing Dependencies

First, create a new project and install MLflow using UV:

```bash
# Initialize UV project
uv init mlflow-tutorial
cd mlflow-tutorial

# Add MLflow and common ML dependencies
uv add mlflow
uv add scikit-learn pandas numpy matplotlib seaborn
uv add jupyter notebook ipykernel
uv add boto3  # For S3 artifact storage (optional)
```

### Environment Setup

Create a `.env` file for configuration:

```bash
# .env
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns
```

## Core Concepts

### 1. Experiments
- Logical grouping of runs
- Default experiment created automatically
- Custom experiments for different projects/models

### 2. Runs
- Individual execution of ML code
- Tracks parameters, metrics, artifacts, and metadata
- Can be nested for complex workflows

### 3. Parameters
- Input values to your ML code (hyperparameters)
- Key-value pairs (strings)
- Immutable during a run

### 4. Metrics
- Quantitative measures of model performance
- Key-value pairs (numeric)
- Can log multiple values for same metric (time series)

### 5. Artifacts
- Output files from runs
- Models, plots, data files, etc.
- Stored in artifact store (local filesystem, S3, etc.)

## Basic Usage Patterns

### 1. Simple Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 2. Experiment Management

```python
# Create or get experiment
experiment_name = "iris-classification"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# Set experiment
mlflow.set_experiment(experiment_name)
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

with mlflow.start_run(run_name="hyperparameter_tuning"):
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            with mlflow.start_run(nested=True):
                mlflow.log_params({
                    "n_estimators": n_est,
                    "max_depth": depth
                })
                
                model = RandomForestClassifier(
                    n_estimators=n_est, 
                    max_depth=depth
                )
                model.fit(X_train, y_train)
                
                accuracy = accuracy_score(y_test, model.predict(X_test))
                mlflow.log_metric("accuracy", accuracy)
                mlflow.sklearn.log_model(model, "model")
```

### 4. Model Registration

```python
# Register model during logging
with mlflow.start_run():
    # ... training code ...
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="iris_classifier"
    )

# Or register existing model
model_uri = "runs:/{}/model".format(run_id)
mlflow.register_model(model_uri, "iris_classifier")
```

### 5. Model Versioning and Staging

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model to staging
client.transition_model_version_stage(
    name="iris_classifier",
    version=1,
    stage="Staging"
)

# Promote to production
client.transition_model_version_stage(
    name="iris_classifier",
    version=1,
    stage="Production"
)
```

## Advanced Features

### 1. Custom Metrics and Artifacts

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

with mlflow.start_run():
    # ... model training ...
    
    # Log confusion matrix as artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    # Log custom metrics
    mlflow.log_metrics({
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted')
    })
```

### 2. Autologging

```python
# Enable autologging for sklearn
mlflow.sklearn.autolog()

# Now all sklearn operations are automatically logged
model = RandomForestClassifier()
model.fit(X_train, y_train)  # Automatically logged!
```

### 3. Model Serving

```bash
# Serve model locally
mlflow models serve -m "models:/iris_classifier/Production" -p 5000

# Serve specific run
mlflow models serve -m "runs:/<run_id>/model" -p 5000
```

### 4. Docker Deployment

```bash
# Build Docker image
mlflow models build-docker -m "models:/iris_classifier/Production" -n my-model

# Run container
docker run -p 5000:8080 my-model
```

## MLflow UI

Start the MLflow UI to visualize experiments:

```bash
uv run mlflow ui
```

Access at `http://localhost:5000` to:
- Compare experiments and runs
- Visualize metrics over time
- Download artifacts
- Manage model registry

## Best Practices

### 1. Experiment Organization

```python
# Use descriptive experiment names
mlflow.set_experiment("customer-churn-prediction-v2")

# Use run names for clarity
with mlflow.start_run(run_name="baseline-logistic-regression"):
    pass
```

### 2. Consistent Logging

```python
# Always log key information
with mlflow.start_run():
    # Data info
    mlflow.log_param("dataset_size", len(X_train))
    mlflow.log_param("feature_count", X_train.shape[1])
    
    # Model config
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_params(model.get_params())
    
    # Performance
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "training_time": training_time
    })
```

### 3. Artifact Management

```python
# Save preprocessing artifacts
with mlflow.start_run():
    # Save scaler
    import joblib
    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")
    
    # Save feature names
    with open("features.txt", "w") as f:
        f.write("\n".join(feature_names))
    mlflow.log_artifact("features.txt")
```

### 4. Environment Reproducibility

Create `MLproject` file:

```yaml
name: iris-classification

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
    command: "python train.py {n_estimators} {max_depth}"
```

## Integration Examples

### 1. With Jupyter Notebooks

```python
# Set tracking URI in notebook
import mlflow
mlflow.set_tracking_uri("file:./mlruns")

# Use experiment context
with mlflow.start_run():
    # Your notebook code here
    pass
```

### 2. With Hyperopt

```python
from hyperopt import fmin, tpe, hp, Trials

def objective(params):
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        # Train model with params
        # Return metric to minimize
        return {'loss': error, 'status': STATUS_OK}

trials = Trials()
best = fmin(
    fn=objective,
    space={
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [5, 10, 15])
    },
    algo=tpe.suggest,
    max_evals=20,
    trials=trials
)
```

### 3. Production Deployment

```python
# Load production model
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("models:/iris_classifier/Production")

# Make predictions
predictions = model.predict(new_data)
```

## Troubleshooting

### Common Issues

1. **Permission errors**: Check file permissions in mlruns directory
2. **Database locks**: Use different tracking URIs for concurrent runs
3. **Large artifacts**: Consider using remote storage (S3, Azure Blob)
4. **Memory issues**: Use chunked logging for large datasets

### Performance Tips

1. **Batch logging**: Use `mlflow.log_metrics()` instead of multiple `log_metric()`
2. **Artifact compression**: Compress large artifacts before logging
3. **Selective logging**: Don't log everything, focus on important metrics
4. **Clean up**: Regularly delete old experiments and runs

## Project Structure

```
mlflow-tutorial/
├── pyproject.toml          # UV configuration
├── .env                    # Environment variables
├── notebooks/              # Jupyter notebooks
│   └── mlflow_examples.ipynb
├── src/                    # Source code
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── utils.py           # Utility functions
├── data/                  # Data files
├── models/                # Saved models
├── mlruns/                # MLflow tracking data
└── MLproject              # MLflow project file
```

This guide provides a comprehensive foundation for using MLflow in your data science projects. The accompanying Jupyter notebook will give you hands-on experience with these concepts. 