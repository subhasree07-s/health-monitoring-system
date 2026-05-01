from src.train import train_model
from src.evaluate import evaluate_model

def main():
    try:
        print("🚀 Starting model training...")
        model, X_test, y_test = train_model()

        print("\n📊 Evaluating model performance...")
        evaluate_model(model, X_test, y_test)

        print("\n✅ Pipeline executed successfully!")

    except Exception as e:
        print("❌ Error occurred:", e)

if __name__ == "__main__":
    main()