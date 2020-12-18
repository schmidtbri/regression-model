import unittest

from regression_model.prediction.model import InsuranceChargesModel
from regression_model.prediction.schemas import InputSchema, OutputSchema, SexEnum, RegionEnum
from ml_base.ml_model import MLModelSchemaValidationException


class ModelTests(unittest.TestCase):

    def test_model(self):
        # arrange
        model = InsuranceChargesModel()
        inpt = dict(age=35, sex=SexEnum.male, bmi=20.0, children=1, smoker=False, region=RegionEnum.northeast)

        # act
        prediction = model.predict(inpt)

        # assert
        self.assertTrue(type(prediction) is OutputSchema)

    def test_model_with_wrong_input_type(self):
        # arrange
        model = InsuranceChargesModel()
        inpt = dict(age=35.0, sex="asdas", bmi=20.0, children=1, smoker=False, region=RegionEnum.northeast)

        # act, assert
        with self.assertRaises(MLModelSchemaValidationException):
            prediction = model.predict(inpt)
            

if __name__ == '__main__':
    unittest.main()
