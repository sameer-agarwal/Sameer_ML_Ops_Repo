import sys
import torch
sys.path.insert(1,r'C:\Users\Sameer Agarwal\OneDrive - Danmarks Tekniske Universitet\Documents\Courses\DTU ML OPs\04_Continuous_Integration\Structure_Example_repo\src\models')
from model import MyAwesomeModel
import pytest

@pytest.mark.parametrize("x", [torch.rand(1,2,4,3), torch.rand(1,2,3,3), torch.rand(1,23,4,3)])

def test(x):
    with pytest.raises(ValueError,match = r"Expected .*"):
        test_call(x)

def test_call(x):

    model_test = MyAwesomeModel()
    model_test.forward(x)