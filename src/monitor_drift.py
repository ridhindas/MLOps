import pandas as pd
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_data_drift():
    # 1. Load reference data (Iris training set)
    iris = load_iris(as_frame=True)
    reference = iris.frame.sample(n=100, random_state=42)

    # 2. Simulate current production data with shifted distributions
    current = iris.frame.sample(n=100, random_state=99)
    current['sepal length (cm)'] = current['sepal length (cm)'] * 1.5  # Artificial drift

    # 3. Generate Evidently Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)

    # 4. Save visual report and extract metric results
    drift_report.save_html("drift_report.html")
    report_dict = drift_report.as_dict()
    
    # Extract overall dataset drift flag
    dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
    print(f"Dataset Drift Detected: {dataset_drift}")
    
    return dataset_drift

if _name_ == "_main_":
    check_data_drift()
