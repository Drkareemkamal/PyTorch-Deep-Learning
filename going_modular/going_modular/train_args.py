
import torch
import os
import argparse

from torchvision import transforms
import data_setup , engine , model_builder , utils

parser = argparse.ArgumentParser(description='Get Some Hyperparameters.')

# Get an arg from num_epochs
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="The number of epochs to train for")

parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="the number of samples per batch")

parser.add_argument("--hidden_units",
                    default=10,
                    type= int,
                    help="the number of hidden units in hidden layers")

parser.add_argument("--learning_rate",
                    default = 0.001,
                    type=float,
                    help = 'learning rate to use for the model')

# Create an arg for train directory
parser.add_argument("--train_dir",
                    default="/content/drive/MyDrive/PyTorch/pizza_steak_sushi/data/pizza_steak_sushi/train",
                    type= str,
                    help = "directory file path to training data in standard image classification format")

# Create an arg for test directory
parser.add_argument("--test_dir",
                    default="/content/drive/MyDrive/PyTorch/pizza_steak_sushi/data/pizza_steak_sushi/test",
                    type= str,
                    help = "directory file path to testing data in standard image classification format")

# Get our arguments from parser
args = parser.parse_args()

# Setup Hyperparameters

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
HIDDIN_UNITS = args.hidden_units

print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDIN_UNITS} and learning rate : {LEARNING_RATE}")


# setup directory

train_dir = args.train_dir
test_dir = args.test_dir

print(f"[INFO] Training data file : {train_dir}")
print(f"[INFO] Testing data file : {test_dir}")

# setup target device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# setup transforms

data_transform = transforms.Compose([
                transforms.Resize(size=(64,64)),
                transforms.ToTensor()
])

# setup dataloader

train_dataloader , test_dataloader , class_names = data_setup.create_dataloader(train_dir = train_dir,
                                                                                test_dir = test_dir,
                                                                                transform = data_transform,
                                                                                batch_size=BATCH_SIZE,)

# create a model
model_3 = model_builder.TinyVGG(input_shape = 3,
                                hidden_units= HIDDIN_UNITS,
                                output_shape = len(class_names)).to(device)

# create a loss and optimizer function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model_3.parameters(),
                              lr = LEARNING_RATE)

# train the model
results = engine.train(model= model_3,
                        train_dataloader= train_dataloader,
                        test_dataloader = test_dataloader,
                        epochs = NUM_EPOCHS,
                        loss_fn = loss_fn,
                        optimizer = optimizer,
                        device=device)

# save model 
utils.save_model(model = model_3,
                target_dir = 'models_args',
                model_name= "6_going_modular_tinyvgg_model.pth")
