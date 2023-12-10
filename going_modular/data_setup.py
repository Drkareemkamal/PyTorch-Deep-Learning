"""
Contains functionality for creating PyTorch Dataloaders dor Omage classification dada
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets , transforms

NUM_WORKERS = os.cpu_count()

def create_dataloader(train_dir:str,
                      test_dir:str,
                      transform : transforms.Compose,
                      batch_size:int,
                      num_workers :int=NUM_WORKERS):
  """ Creates training and testing DataLoader.

  Takes in a training directory and testing directory path and turn into
  PyTorch Datasets and then into a PyTorch DataLoaders.

  Args :
    train_dir : Path to training directory
    test_dir : Path to testing directory
    transform : torchvision transforms to perform on training and testing data.
    batch_size : Number of samoles per batch in each of the DataLoaders.
    num_workers : An intger for Number of workers per Dataloader

  Returns :
    A tuple of (train_dataloader , test_dataloader, class_names)
    Where class_names is a list of target classes.
      Example usage
        train_dataloader , test_dataloader , class_names = create_dataloaders(train_dir=path/to/train_dir,
        test_dir= path/to/test_dir,
        transform = some_transform,
        batch_size = 32,
        num_workers = 4
        )


  """
  # USe ImageFolder to create dataset(s)

  train_imgs = datasets.ImageFolder(root=train_dir,
                          transform=transform)
  
  test_imgs = datasets.ImageFolder(root=test_dir,
                          transform=transform)

  # Get the class names 
  class_names = train_imgs.classes

  # Turn an images into DataLoader 
  train_dataloader = DataLoader(dataset=train_imgs,
                                batch_size=batch_size,
                                num_workers=NUM_WORKERS,
                                shuffle= True,
                                pin_memory=True)

  test_dataloader = DataLoader(dataset= test_imgs,
                                batch_size=batch_size,
                                num_workers=NUM_WORKERS,
                                shuffle= False,
                                pin_memory= True)
  return train_dataloader ,test_dataloader,  class_names
