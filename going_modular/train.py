"""
Train a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from torchvision import transforms 
import data_setup , engine , model_builder , utils

# setup hyperparameters

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDIN_UNITS = 10
LEARNING_RATE = 0.001

# Setpu directories
train_dir = "/content/drive/MyDrive/PyTorch/pizza_steak_sushi/data/pizza_steak_sushi/train"
test_dir = '/content/drive/MyDrive/PyTorch/pizza_steak_sushi/data/pizza_steak_sushi/test'

# setup device agnositc code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create a transforms 
data_transform = transforms.Compose([
  transforms.Resize(size=(64,64)),
  transforms.ToTensor()
])

# create DataLoader's and get class names
train_dataloader , test_dataloader , class_names = data_setup.create_dataloader(train_dir = train_dir,
                                                                                test_dir = test_dir,
                                                                                transform= data_transform,
                                                                                batch_size= BATCH_SIZE)

# create the model 

model_2 = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=HIDDIN_UNITS,
                                  output_shape=len(class_names)).to(device)

# setup loss and optimizer

loss_fn  = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model_2.parameters(),
                             lr= LEARNING_RATE)

# start the timer

from timeit import default_timer as timer
start_time = timer()

# start training with help from engine.py

engine.train(model=model_2,
            train_dataloader = train_dataloader,
            test_dataloader = test_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            epochs = NUM_EPOCHS,
            device=device)

end_time = timer()
print(f"[INFO] Total training time : {end_time - start_time:.3f} seconds")

# save the model 

utils.save_model(model=model_2,
                target_dir= ' models',
                model_name= '6_Pytorch_going_modular.pth')
