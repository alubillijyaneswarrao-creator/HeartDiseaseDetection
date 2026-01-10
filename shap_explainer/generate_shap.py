import joblib
import shap
import pandas as pd
import numpy as np

# ===============================
# STEP 1: Load trained pipeline
# ===============================
model = joblib.load("../backend/artifacts/svm_model.pkl")

# ===============================
# STEP 2: Load processed dataset
# ===============================
df = pd.read_csv("heart_processed.csv")
X = df.drop("target", axis=1)

# ===============================
# STEP 3: Apply same scaling
# ===============================
X_scaled = model.named_steps["scaler"].transform(X)

# ===============================
# STEP 4: Create SHAP explainer (ONCE)
# ===============================
explainer = shap.Explainer(
    model.named_steps["svm"],
    X_scaled
)

# ===============================
# STEP 5: Compute SHAP (sample only)
# ===============================
shap_values = explainer(X_scaled[:50])  # ✅ best practice

# ===============================
# STEP 6: Save SHAP outputs
# ===============================
joblib.dump(shap_values, "shap_values.pkl")
joblib.dump(list(X.columns), "shap_feature_names.pkl")

print("✅ SHAP values generated and saved successfully")
