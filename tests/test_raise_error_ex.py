#import sys
import torch
#sys.path.insert(1,r'C:\Users\Sameer Agarwal\OneDrive - Danmarks Tekniske Universitet\Documents\Courses\DTU ML OPs\DTU_ML_OPs\Sameer_ML_Ops_Repo\src\models')
from src.models.model import MyAwesomeModel
import pytest    


@pytest.mark.parametrize("x", [torch.rand([1,2,4,3]), torch.rand([1,2,3,3]), torch.rand([1,23,4,3])])
def test(x):
    with pytest.raises(ValueError,match = r"Expected .*"):
        f(x)

def f(x):
    model_test = MyAwesomeModel()
    model_test.forward(x)   
