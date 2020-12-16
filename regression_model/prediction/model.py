from ml_base import MLModel

from regression_model import __version__
from regression_model.prediction.schemas import InputSchema, OutputSchema


class HouseValueModel(MLModel):
    @property
    def display_name(self) -> str:
        return "house_value_model"

    @property
    def qualified_name(self) -> str:
        return "House Value Model"

    @property
    def description(self) -> str:
        return "Model to predict the value of a house based on its features."

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
        pass

    def predict(self, data):
        return OutputSchema()
