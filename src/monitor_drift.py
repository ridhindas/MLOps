import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset

def check_data_drift():
    iris = load_iris(as_frame=True)
    reference = iris.frame.sample(n=100, random_state=42)
    current = iris.frame.sample(n=100, random_state=99)
    current['sepal length (cm)'] = current['sepal length (cm)'] * 1.5

    try:
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        snapshot.save_html("drift_report.html")
    except Exception as e:
        print(f"HTML Report rendering skipped: {e}")

    drifted_columns = 0
    for col in iris.feature_names:
        p_val = ks_2samp(reference[col], current[col]).pvalue
        if p_val < 0.05:
            drifted_columns += 1

    dataset_drift = drifted_columns >= 1
    print(f"Dataset Drift Detected: {dataset_drift}")
    return dataset_drift

if __name__ == "__main__":
    check_data_drift()
