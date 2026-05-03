from src.evaluation.metrics import evaluate


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    results = evaluate(y_test, y_pred)

    print("Evaluation results:", results)

    return results