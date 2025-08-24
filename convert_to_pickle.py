import joblib
import pickle

# Load existing joblib files
model = joblib.load("model.pkl")
feature_info = joblib.load("feature_info.pkl")
metrics = joblib.load("metrics.pkl")

# Re-save with pickle
with open("model_pickle.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_info_pickle.pkl", "wb") as f:
    pickle.dump(feature_info, f)

with open("metrics_pickle.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("All files converted to pickle format!")
