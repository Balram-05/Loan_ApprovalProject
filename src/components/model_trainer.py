import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report


class ModelTrainer:

    def TrainModel(self, X, y, preprocessor):

        # 1. Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. Define pipeline
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", GradientBoostingClassifier(random_state=42))
            ]
        )

        # 3. Hyperparameter grid
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }

        print("Starting Hyperparameter Tuning with Gradient Boosting...")

        # 4. GridSearch
        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )

        # 5. Train model
        grid_search.fit(X_train, y_train)

        # 6. Best model
        best_model = grid_search.best_estimator_
        print(f"Best Parameters found: {grid_search.best_params_}")

        # 7. Evaluate model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("\n--- Model Evaluation (Gradient Boosting) ---")
        print("Best Model Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # 8. Get feature names after preprocessing
        feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

        # 9. Get feature importance from model
        importance = best_model.named_steps["model"].feature_importances_

        # 10. Create dataframe
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        # 11. Save feature importance
        os.makedirs("models", exist_ok=True)
        feature_importance_df.to_csv("models/feature_importance.csv", index=False)

        print("Feature importance saved!")

        # 12. Save trained model
        joblib.dump(best_model, "models/model.pkl")

        print("Gradient Boosting model saved successfully!")

        return best_model