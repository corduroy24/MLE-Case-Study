from config import XGBOOST_PARAMS, MODEL_PATH, METRICS_PATH
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import joblib
import json

def train_classifier(X_train, y_train, X_test, y_test):
    xgb_clf = xgb.XGBClassifier(
            **XGBOOST_PARAMS, 
            objective="binary:logistic", 
            eval_metric="auc", 
            tree_method="hist"
        )
    
    xgb_clf.fit(X_train, y_train)

    y_pred = xgb_clf.predict(X_test)
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]


    joblib.dump(xgb_clf, MODEL_PATH)   # save model


    # evaluation metrics
    metrics = {
        "auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(), 
    }
    print('metrics: ', metrics)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)


    return xgb_clf

