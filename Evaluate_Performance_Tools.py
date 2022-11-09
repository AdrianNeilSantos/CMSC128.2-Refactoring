from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)

def evaluate_model(model, model_name, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"{model_name} Accuracy", accuracy_score(y_test, y_pred))
    print(f"Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print(f"Classification Report {model_name}\n", classification_report(y_test, y_pred))
