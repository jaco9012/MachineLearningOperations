import argparse
import sys
import torch
from torch import nn, optim
from torchvision import datasets
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from src.data.dataset import mnist
from models.model import MyAwesomeModel


class predict(object):
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
    
    def predict(self):
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
        testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

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
        
        # Plot the image and prediction      
        org_images = images.resize_(1, 28, 28)
        
        plt.imshow(org_images.permute(1,2,0)[:,:,0])
        plt.xlabel('Predicted: %s' % top_class.item() + ' - True label: %s' % labels.item())
        plt.savefig("reports/figures/prediction.png")

if __name__ == '__main__':
    predict()
    