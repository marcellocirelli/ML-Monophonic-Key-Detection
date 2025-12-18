import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Loads the dataset
data = torch.load("chroma_tensors.pt")
X_pt = data["X"]
Y_pt = data["y"]

# Converts tensor to NumPy array
X = np.stack(
    [x.numpy().reshape(-1) for x in X_pt]
)
y = np.array(Y_pt)

# Splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Initializes cross-validation object
cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)

# Model definition
rfc = RandomForestClassifier(n_jobs = -1, random_state = 42)
hgbc = HistGradientBoostingClassifier(random_state = 42)

# Trains the models
print("Fitting Random Forest...")
rfc.fit(X_train, y_train)
print("Completed fitting Random Forest!")
print("Fitting HistGradientBoosting...")
hgbc.fit(X_train, y_train)
print("Completed fitting HistGradientBoosting!\n")

# Optimizes using RandomizedSearchCV
def tune_randomized(estimator, param_dist, X_train, y_train, cv):
    search = RandomizedSearchCV(
        estimator = estimator,
        param_distributions = param_dist,
        scoring = "accuracy",
        cv = cv,
        n_iter = 5,
        n_jobs = -1,
        random_state = 42,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    return best

# Hyperparameter tuning
rfc_param_dist = {
    "n_estimators": [100, 150],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# !! Attempting to fit the tuned HistGradientBoosting model would take 4 hours
# hgbc_param_dist = {
#     "learning_rate": [0.05, 0.1, 0.2],
#     "max_iter": [50, 100],
#     "max_depth": [None, 5, 10],
# }

# Tuned models
rfc_tuned = tune_randomized(rfc, rfc_param_dist, X_train, y_train, cv)
# hgbc_tuned = tune_randomized(hgbc, hgbc_param_dist, X_train, y_train, cv)

# Metric report function
def report_metrics(model, X_t, y_t, name: str):
    y_pred = model.predict(X_t)
    print(name)
    accuracy = accuracy_score(y_t, y_pred)
    precision = precision_score(y_t, y_pred, average = "weighted")
    recall = recall_score(y_t, y_pred, average = "weighted")
    f1 = f1_score(y_t, y_pred, average = "weighted")
    print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n")
    print("Confusion Matrix:")
    print(f"{confusion_matrix(y_t, y_pred)}\n\n")

# Prints metrics
models_to_eval = [
    ("RandomForest Baseline", rfc),
    ("HistGradientBoosting Baseline", hgbc),
    ("RandomForest Tuned", rfc_tuned),
    # ("HistGradientBoosting Baseline", hgbc),
]

print("Metrics:")
for name, model in models_to_eval:
    report_metrics(model, X_test, y_test, name)
