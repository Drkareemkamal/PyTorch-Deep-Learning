
from pathlib import Path
import torch

def save_model(model : torch.nn.Module,
               target_dir : str,
               model_name : str):
  
  """ Saves a PyTorch Model to a target directory

  Args :
    model : A target PyTorch model to save
    target_dit : A directory for saving the model to.
    model_name : A file name for the saving model. should include
      either ".pth" or ".pt" as the file extension.

  Example usage :
    save_model(model=model_0,
              target_dir='models',
              model_names = "05_going_modular_tintvgg_model.pth)
  """
  

  #create a target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith('.pt') , 'model_name should end with ".pt" or ".pt '
  model_save_path = target_dir_path / model_name

  # save the model state dict
  print(f"[INFO] saving model to : {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
