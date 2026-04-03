"""
evaluate.py
-----------
Loads ARIMA and LSTM result JSONs, compares RMSE values,
declares the winning model, and saves a comparison report.
"""

import json
import os


def compare_models(output_dir: str = "outputs") -> dict:
    """
    Read previously saved result files and pick the better model.

    Parameters
    ----------
    output_dir : str
        Root output folder (must already contain predictions/arima_results.json
        and predictions/lstm_results.json).

    Returns
    -------
    dict  {arima_rmse, lstm_rmse, best_model, improvement_pct}
    """
    pred_dir = os.path.join(output_dir, "predictions")

    arima_path = os.path.join(pred_dir, "arima_results.json")
    lstm_path  = os.path.join(pred_dir,  "lstm_results.json")

    if not os.path.exists(arima_path):
        raise FileNotFoundError(f"ARIMA results not found: {arima_path}")
    if not os.path.exists(lstm_path):
        raise FileNotFoundError(f"LSTM results not found: {lstm_path}")

    with open(arima_path) as f:
        arima_results = json.load(f)
    with open(lstm_path) as f:
        lstm_results = json.load(f)

    arima_rmse = arima_results["rmse"]
    lstm_rmse  = lstm_results["rmse"]

    # Lower RMSE = better
    if arima_rmse <= lstm_rmse:
        best_model = "ARIMA"
        worse_rmse = lstm_rmse
    else:
        best_model = "LSTM"
        worse_rmse = arima_rmse

    best_rmse = min(arima_rmse, lstm_rmse)
    improvement_pct = round((worse_rmse - best_rmse) / worse_rmse * 100, 2)

    print("\n" + "=" * 50)
    print("         MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(f"  ARIMA RMSE : {arima_rmse:.4f}")
    print(f"  LSTM  RMSE : {lstm_rmse:.4f}")
    print(f"  ✅ Best Model : {best_model}  "
          f"(↓ {improvement_pct} % lower RMSE)")
    print("=" * 50 + "\n")

    comparison = {
        "arima_rmse": arima_rmse,
        "lstm_rmse": lstm_rmse,
        "best_model": best_model,
        "improvement_pct": improvement_pct,
    }

    # Persist comparison report
    report_path = os.path.join(pred_dir, "model_comparison.json")
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[Evaluate] Comparison report saved → {report_path}")

    return comparison
