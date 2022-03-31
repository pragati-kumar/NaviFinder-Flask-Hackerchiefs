import unittest
import os
import sys
sys.path.append('../')
from mlScripts.indoor import getCoordinates


class TestNavigation(unittest.TestCase):

    def test(self):
        self.model_path = '../mlScripts/learn_model_dict0.pth'
        self.scaler_path = '../mlScripts/ss_scaler.save'

        self.start_output = [None, None, None]
        self.user_id = '12344'
        if os.path.exists(f'{self.user_id}.pkl'):
            os.remove(f'{self.user_id}.pkl')

        # First we should get None in start in output for 5 calls
        self.assertListEqual(getCoordinates(self.user_id, -61, self.model_path, self.scaler_path),
                             self.start_output, 'Starting output wrong')
        self.assertListEqual(getCoordinates(
            self.user_id, -61, self.model_path, self.scaler_path, trial=False), self.start_output, 'Starting output wrong')
        self.assertListEqual(getCoordinates(self.user_id, -61, self.model_path, self.scaler_path, trial=False),
                             self.start_output, 'Starting output wrong')
        self.assertListEqual(getCoordinates(self.user_id, -61, self.model_path, self.scaler_path, trial=False),
                             self.start_output, 'Starting output wrong')
        self.assertListEqual(getCoordinates(self.user_id, -61, self.model_path, self.scaler_path, trial=False),
                             self.start_output, 'Starting output wrong')
        self.assertListEqual(getCoordinates(self.user_id, -61, self.model_path, self.scaler_path, trial=False),
                             self.start_output, 'Starting output wrong')

        #Now checking if resulting in different rssi(signal strength at diferent location) should result in different x,y
        first_pred = getCoordinates(
            self.user_id, -61, self.model_path, self.scaler_path, trial=False)
        second_pred = getCoordinates(
            self.user_id, -62, self.model_path, self.scaler_path, trial=False)

        assert first_pred != second_pred


if __name__ == '__main__':
    unittest.main()
