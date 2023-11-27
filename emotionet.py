import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd

# Try to use CUDA for the training to reduce times
cpu_device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class EmotioNet(nn.Module):

    def __init__(self):
        super().__init__()

        # The Convolution layers to use for Image classification
        self.con_layers = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),

                        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                    )

        # The linear layers to use to reduce the conv into its parameters
        self.linear_layers = nn.Sequential(
                                            nn.Flatten(),
                                            nn.Linear(512 * 3 * 3, 1024),
                                            nn.ReLU(),

                                            nn.Linear(1024, 512),
                                            nn.ReLU(),

                                            nn.Dropout(0.8), # Apply a dropout to reduce overfitting
                                            nn.Linear(512, 3)
                                        )

    def forward(self, input):
        # View the input as a 48x48 image and give it to the conv layers, then
        # the linear layers
        input = input.view(input.size(0), 1, 48, 48)

        input = self.con_layers(input)
        input = self.linear_layers(input)

        return input

def training_loop(
                    epochs: int,
                    optimizer: optim.Optimizer,
                    model: nn.Module,
                    loss_fn: nn.MSELoss,
                    train_loader: DataLoader,
                ) -> None:
    """
    Does the main training loop given the number of epochs, an optimizer,
    a model, and a DataLoader that loads the training data
    """

    for epoch in range(1, epochs + 1):

        # Maintain a list of losses to calculate the average loss at
        # certain points
        losses = []
        for train_data, train_target in train_loader:
            model.train() # Use training mode to apply the dropout layer

            # Retrieve our batches of training data and load them to CUDA
            train_data, train_target = train_data.to(device), train_target.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Find a prediction and its loss from the training data
            train_prediction = model(train_data)
            train_loss = loss_fn(train_prediction, train_target)
            losses.append(train_loss)

            # Preform gradient descent and backwards propogation
            train_loss.backward()
            optimizer.step()

        # Print diagnostics info every 2 epochs
        if epoch == 1 or not epoch % 2:
            print(f"Average Loss @ {epoch}: {sum(losses)/len(losses)}")

# DATA LOADING
train_data_raw = pd.read_csv('train_data.csv', header=None).to_numpy()
train_target_raw = pd.read_csv('train_target.csv', header=None).to_numpy().flatten()

train_data = torch.from_numpy(train_data_raw).type(torch.float32).view(-1, 2304)
train_target = torch.from_numpy(train_target_raw).type(dtype=torch.long)

train_loader = DataLoader(TensorDataset(train_data, train_target), batch_size=32, shuffle=True)

# Create the model and the optimizer
emotionet = EmotioNet().to(device)
optimizer = optim.Adam(emotionet.parameters(), lr=0.0001)

# Run the training loop
training_loss = training_loop(
                                epochs = 7,
                                optimizer = optimizer,
                                model = emotionet,
                                loss_fn = nn.CrossEntropyLoss(),
                                train_loader=train_loader,
                            )

emotionet.eval() # Use eval mode to stop using dropout layer

# Load the test data to the device
test_data_raw = pd.read_csv('test_data.csv', header=None).to_numpy()
test_data = torch.from_numpy(test_data_raw).type(torch.float32).view(-1, 2304)

# Make a prediction and retrieve the chosen prediction classes
# Use CPU since we aren't doing a training loop, no need to use GPU
test_prediction = emotionet.to(cpu_device)(test_data)
pred_classes = torch.argmax(test_prediction, dim=1)

# Write the data to CSV
dataframe = pd.DataFrame(
                            {
                                'Id': list(range(len(pred_classes))),
                                'Category': pred_classes.tolist()
                            }
                    )
dataframe.to_csv("submission.csv", index=False)