import sys
import argparse
import click

import torch
from torch import nn,optim

from model import MyAwesomeModel

import numpy as np

@click.command()
@click.argument('training_filepath', type=click.Path(exists=True))

def main(training_filepath):

        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        CheckPoint = torch.load(r'C:\Users\Sameer Agarwal\OneDrive - Danmarks Tekniske Universitet\Documents\Courses\DTU ML OPs\02_Code_Organisation\Structure_Example_repo\src\models\Saved_Models\checkpoint.pth')
        
        model.load_state_dict(CheckPoint)
        model.eval()
        
        test_set = training_filepath
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        
        equals_list = []
        
        with torch.no_grad():
        # validation pass here
            for images, labels in test_set:
                # Get the class probabilities
                ps = torch.exp(model(images))
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                equals_list.append(equals.numpy())
            # Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
    
            equals_list = np.array(equals_list)
            print(equals_list.shape)
            equals_list = equals_list*1
            accuracy = equals_list[0].mean()
            print(f'Accuracy: {accuracy*100}%')
            return accuracy

        
if __name__ == '__main__':
    main(training_filepath)
    


        