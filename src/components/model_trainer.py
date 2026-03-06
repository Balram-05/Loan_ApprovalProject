import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:

    def TrainModel(self, X, y, preprocessor):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # full pipeline (preprocessing + model)
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(random_state=42))
            ]
        )

        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("Model Accuracy:", accuracy)

        # create models folder
        os.makedirs("models", exist_ok=True)

        # save model
        joblib.dump(model_pipeline, "models/model.pkl")

        print("Model saved successfully!")

        return model_pipeline