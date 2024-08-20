import numpy as np

def profit_metric(y_true, y_pred_proba):
    # Classify as "Yes" if the predicted probability is greater than 0.08
    y_pred = (y_pred_proba > 0.0833).astype(int)  # 1 for "Yes", 0 for "No"
    
    # Calculate profit based on the classification and the true label
    profit = np.where((y_pred == 1) & (y_true == 1), 165, 0)  # Profit when correctly classified as "Yes"
    profit += np.where((y_pred == 1) & (y_true == 0), -15, 0)  # Loss when incorrectly classified as "Yes"
    
    # Return the sum of the profit as a Python integer
    return profit.mean()