from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainPipeline:

    def run_pipeline(self):

        # Step 1 : Data Ingestion
        ingestion = DataIngestion()
        data = ingestion.Ingest_data("C:\\Users\\chara\\OneDrive\\Desktop\\Loan Approval Project\\data\\loan_dataset.csv")

        print("Data Ingestion Completed")

        # Step 2 : Data Preprocessing
        preprocessing = DataPreprocessing()
        X, y, preprocessor = preprocessing.PreProcessData(data)

        print("Data Preprocessing Completed")

        # Step 3 : Model Training
        trainer = ModelTrainer()
        trainer.TrainModel(X, y, preprocessor)

        print("Model Training Completed")


if __name__ == "__main__": #jab tak main function me na ho, tab tak run ni krna

    pipeline = TrainPipeline()
    pipeline.run_pipeline()