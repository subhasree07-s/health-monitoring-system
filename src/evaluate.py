import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    print("\nClass-wise Predictions:", set(y_pred))

    # Create folders safely
    os.makedirs("outputs/plots", exist_ok=True)

    # Save metrics
    with open("outputs/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1 Score: {f1}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = model.classes_

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()

    plt.title("Confusion Matrix")

    # Save plot
    plt.savefig("outputs/plots/confusion_matrix.png")
    plt.show()