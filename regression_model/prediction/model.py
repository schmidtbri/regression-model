import os
from ml_base import MLModel
from ml_base.ml_model import MLModelSchemaValidationException
import joblib
import pandas as pd

from regression_model import __version__
from regression_model.prediction.schemas import InputSchema, OutputSchema


class InsuranceChargesModel(MLModel):
    @property
    def display_name(self) -> str:
        return "insurance_charges_model"

    @property
    def qualified_name(self) -> str:
        return "Insurance Charges Model"

    @property
    def description(self) -> str:
        return "Model to predict the insurance charges of a customer.."

    @property
    def version(self) -> str:
        return __version__

    @property
    def input_schema(self):
        return InputSchema

    @property
    def output_schema(self):
        return OutputSchema

    def __init__(self):
        """Class constructor that loads and deserializes the model parameters.

        .. note::
            The trained model parameters are loaded from the "model_files" directory.

        """
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        with open(os.path.join(dir_path, "model_files", "model.joblib"), 'rb') as file:
            self._svm_model = joblib.load(file)

    def predict(self, data: InputSchema) -> OutputSchema:
        """Make a prediction with the model.

        :param data: Data for making a prediction with the model. Object must meet requirements of the input schema.
        :type data: OutputSchema
        :rtype: dict -- The result of the prediction, the output object will meet the requirements of the output schema.

        """
        if type(data) is not InputSchema:
            raise ValueError("Input object is of wrong type.")

        # converting the incoming dictionary into a pandas dataframe that can be accepted by the model
        X = pd.DataFrame([[data.age, data.sex.value, data.bmi, data.children, data.smoker, data.region.value]],
                         columns=["age", "sex", "bmi", "children", "smoker", "region"])

        # making the prediction and extracting the result from the array
        y_hat = float(self._svm_model.predict(X)[0])

        return OutputSchema(charges=y_hat)
