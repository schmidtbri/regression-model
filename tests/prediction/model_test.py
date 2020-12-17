import unittest

from regression_model.prediction.model import InsuranceChargesModel
from regression_model.prediction.schemas import InputSchema, OutputSchema, SexEnum, RegionEnum


class ModelTests(unittest.TestCase):

    def test_model(self):
        # arrange
        model = InsuranceChargesModel()
        input = InputSchema(age=35, sex=SexEnum.male, bmi=20.0, children=1, smoker=False, region=RegionEnum.northeast)

        # act
        prediction = model.predict(input)

        # assert
        self.assertTrue(type(prediction) is OutputSchema)

    def test_model_with_wrong_input_type(self):
        # arrange
        model = InsuranceChargesModel()
        input = dict(age=35, sex=SexEnum.male, bmi=20.0, children=1, smoker=False, region=RegionEnum.northeast)

        # act, assert
        with self.assertRaises(ValueError):
            prediction = model.predict(input)


if __name__ == '__main__':
    unittest.main()
