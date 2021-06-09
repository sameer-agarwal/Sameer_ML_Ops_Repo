import sys
import argparse

import torch
from torch import nn,optim

from model import MyAwesomeModel

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, r'C:\Users\Sameer Agarwal\OneDrive - Danmarks Tekniske Universitet\Documents\Courses\DTU ML OPs\02_Code_Organisation\Structure_Example_repo\src\data')
from make_dataset import main

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        print(args)
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.003)

        epochs = 5
        _, train_set, _ = main()
        train_error = []
        epochs_list = []

        for e in range(epochs):
            running_loss = 0
            model.train()
            for images, labels in train_set:
            # Flatten MNIST images into a 784 long vector
                
                optimizer.zero_grad()
                output = model.forward(images)
  
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
            
            print(loss)
            train_error.append(loss.item())
            epochs_list.append(epochs)
        
        plt.plot(epochs, train_error, label='train error')
        plt.legend()
        plt.savefig(r'C:\Users\Sameer Agarwal\OneDrive - Danmarks Tekniske Universitet\Documents\Courses\DTU ML OPs\02_Code_Organisation\Structure_Example_repo\reports\figures\training_loss.png')

        torch.save(model.state_dict(), r'C:\Users\Sameer Agarwal\OneDrive - Danmarks Tekniske Universitet\Documents\Courses\DTU ML OPs\02_Code_Organisation\Structure_Example_repo\src\models\Saved_Models\checkpoint.pth')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        CheckPoint = torch.load(r'C:\Users\Sameer Agarwal\OneDrive - Danmarks Tekniske Universitet\Documents\Courses\DTU ML OPs\02_Code_Organisation\Structure_Example_repo\src\models\Saved_Models\checkpoint.pth')
        
        model.load_state_dict(CheckPoint)
        model.eval()
        _, _, test_set = main()
    
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

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    