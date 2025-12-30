import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """
    Prediction pipeline that loads model + preprocessor once per process.
    """
    _model = None
    _preprocessor = None

    @classmethod
    def _load_artifacts(cls):
        if cls._model is not None and cls._preprocessor is not None:
            return

        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at: {preprocessor_path}")

        cls._model = load_object(model_path)
        cls._preprocessor = load_object(preprocessor_path)

    def predict(self, features: pd.DataFrame):
        try:
            self._load_artifacts()

            data_scaled = self._preprocessor.transform(features)
            preds = self._model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Collects user input and converts to a DataFrame in the exact feature order.
    """

    FEATURE_ORDER = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
        "reading_score",
        "writing_score",
    ]

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [pd.to_numeric(self.reading_score, errors="coerce")],
                "writing_score": [pd.to_numeric(self.writing_score, errors="coerce")],
            }

            df = pd.DataFrame(data)

            # ensure exact order (important for some pipelines)
            df = df[self.FEATURE_ORDER]
            return df

        except Exception as e:
            raise CustomException(e, sys)
