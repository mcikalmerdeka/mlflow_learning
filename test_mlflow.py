#!/usr/bin/env python3
"""
Quick and Simple MLflow test script to verify setup works correctly.
Run this with: uv run python test_mlflow.py
"""

import mlflow
import mlflow.sklearn
import mlflow.models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

def main():
    print("🚀 Testing MLflow setup...")
    
    # Set tracking URI explicitly
    tracking_uri = "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✅ Set tracking URI: {tracking_uri}")
    
    # Load sample data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Loaded iris dataset: {X.shape}")
    
    # Create experiment
    experiment_name = "iris-classification-tutorial"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing experiment: {experiment_name} (ID: {experiment_id})")
    except Exception as e:
        print(f"❌ Error creating experiment: {e}")
        return False
    
    # Set active experiment
    mlflow.set_experiment(experiment_name)
    
    # Start a run
    try:
        with mlflow.start_run(run_name="test-run") as run:
            print(f"✅ Started run: {run.info.run_id}")
            
            # Train a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate accuracy
            accuracy = model.score(X_test, y_test)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 10)
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model with signature (prevents warnings)
            input_example = X_train[:3]
            mlflow.sklearn.log_model(
                model,
                "model",
                input_example=input_example,
                signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
            )
            
            print(f"✅ Model trained with accuracy: {accuracy:.3f}")
            print(f"✅ Run completed successfully!")
            
    except Exception as e:
        print(f"❌ Error during run: {e}")
        return False
    
    print("\n🎯 MLflow test completed successfully!")
    print("Now you can run: uv run mlflow ui")
    print("Then open: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Test failed. Please check the errors above.")
        exit(1)
    else:
        print("\n✅ All tests passed!") 