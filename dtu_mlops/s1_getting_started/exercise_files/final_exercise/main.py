import argparse
import sys

import torch
from torch import nn, optim
from torchvision import datasets

from data import mnist
from model import MyAwesomeModel


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
        parser.add_argument('--save_model_to', default='trained_model.pt')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        # Source the model
        model = MyAwesomeModel()
        model.train()
        print(model)
        # Load the training data
        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        epochs = 5

        train_losses = []
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                
                # Clear the gradients, do this because gradients are accumulated
                optimizer.zero_grad()

                 # Forward pass, then backward pass, then update weights
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()

                # Take an update step and view the new weights
                optimizer.step()

                # Add the loss
                running_loss += loss.item()
            else:
                print(f"Training loss: {running_loss/len(trainloader)}")
                # Append the running_loss for each epoch
                train_losses.append(running_loss)

        torch.save(model.state_dict(), args.save_model_to)
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        state_dict = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(state_dict)
        print(model)
        # Set model to evaluation mode
        model.eval()
        _, test_set = mnist()
        testloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

        criterion = nn.NLLLoss()
        
        with torch.no_grad():
            running_loss = 0
            for images, labels in testloader:

                log_ps = model(images)
                loss = criterion(log_ps, labels)

                running_loss += loss.item()

                # Get the class probabilities
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
            else:
                print(f'Accuracy: {accuracy.item()*100}%')


if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    