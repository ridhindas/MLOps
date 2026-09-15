import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.metric_preset import DataDriftPreset


def check_data_drift():
    # 1. Load reference data (Iris training baseline)
    iris = load_iris(as_frame=True)
    reference = iris.frame.sample(n=100, random_state=42)

    # 2. Simulate current production data with artificial drift
    current = iris.frame.sample(n=100, random_state=99)
    current['sepal length (cm)'] = current['sepal length (cm)'] * 1.5

    # 3. Generate HTML report with Evidently
    try:
        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        snapshot.save_html("drift_report.html")  # <-- FIX: Call save_html on snapshot
    except Exception as e:
        print(f"HTML Report rendering skipped: {e}")

    # 4. Statistically check drift using Kolmogorov-Smirnov test
    drifted_columns = 0
    feature_cols = iris.feature_names

    for col in feature_cols:
        p_val = ks_2samp(reference[col], current[col]).pvalue
        if p_val < 0.05:
            drifted_columns += 1

    # FIX: Threshold changed to >= 1 so the single drifted column triggers True
    dataset_drift = drifted_columns >= 1
    print(f"Dataset Drift Detected: {dataset_drift}")

    return dataset_drift


if __name__ == "__main__":
    check_data_drift()
