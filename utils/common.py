# Import standard and external libraries
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ensure import ensure_annotations
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)
from tqdm import tqdm
from PIL import Image

# Make reproducable
torch.manual_seed(42)

# Import custom modules
from  .logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """ Reads a YANL file from the file path provided

    Parameters:
    ----------
    filepath (Path):  The path and filename of the YAML file
    
    Returns:
    ---------
    Contents of the YAML file
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'yaml file: {path_to_yaml} loaded successfully')
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError('yaml file is empty')
    except Exception as e:
        logger.info(f"An error occurred: {e}")

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    
    """ Create list of directories

    Parameters:
    ----------
    Path_to_directories (list): list of path of directories
    
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger\
                    .info(f'created directory at: {path}')
    except Exception as e:
        logger.info(f"An error occurred: {e}")

# helper function to get mean and std
def get_mean_std(loader):
    """Computes the mean and standard deviation of image data.

    Parameters
    ----------
    loader: a `DataLoader` object producing tensors of shape [batch_size, channels, pixels_x, pixels_y]

    Returns
    ----------
    The mean of each channel as a tensor
    The standard deviation of each channel as a tensor
    Formatted as a tuple (means[channels], std[channels])
    """

    logger.info('Mean & std calculations......')
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    try:
        for data, _ in tqdm(loader, desc="Computing mean and std", leave=False):
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1
        
        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean**2) ** 0.5
        
        return mean, std

    except Exception as e:
        logger.info(f"An error occurred: {e}")
        return None, None


# Training loop function
def train_model(model, dataloader, epochs, loss_fn, optimizer,
                device='cpu', plot_acc_loss=True):
    """
    Trains the model over multiple epochs

    Parameters
    ----------
    model : CNN model to train
    dataloader : DataLoader object for training
    epochs : Number of epochs
    loss_fn : Loss function
    optimizer : Optimizer function
    device : 'cpu' (default) or 'cuda'

    Returns
    ----------
    loss_value : Average loss during training
    train_losses : Loss per epoch
    train_accuracies : Accuracy per epoch
    """
    try:
        train_losses = []
        train_accuracies = []

        for epoch in range(epochs):
            running_loss = 0.0
            average_loss = 0.0
            total_correct = 0

            for i, data in enumerate(tqdm(dataloader,
                                          desc="Training Progress")):
                try:
                    inputs, labels = data

                    # Move tensors to the correct device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                    # Backward pass + optimization
                    loss.backward()
                    optimizer.step()

                    # Compute loss
                    running_loss += loss.item()
                    average_loss += loss.data.item() * inputs.size(0)

                    # Compute accuracy
                    is_correct = torch.eq(torch.argmax(outputs, dim=1), labels)
                    total_correct += torch.sum(is_correct).item()

                except Exception as e:
                    logger.info(f"Error in batch {i}: {str(e)}")

            # Compute epoch loss and accuracy
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_accuracy = 100 * total_correct / len(dataloader.dataset)

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            logger.info(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

        loss_value = average_loss / len(dataloader.dataset)

        # Plot results if required
        if plot_acc_loss:
            try:
                fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharex=True)
                ax[0].plot(train_losses, color='orange')
                ax[0].set_title('Loss per Epoch')
                ax[0].set_ylabel('Loss')
                ax[0].set_xlabel('Epochs')

                ax[1].plot(train_accuracies, color='orange')
                ax[1].set_title('Accuracy per Epoch')
                ax[1].set_ylabel('Accuracy (%)')
                ax[1].set_xlabel('Epochs')

                plt.show()
            except Exception as e:
                logger.info(f"Error plotting accuracy/loss: {str(e)}")

        return loss_value, train_losses, train_accuracies

    except Exception as e:
        logger.info(f"Error in train_model: {str(e)}")
        return None, None, None



# Evaluation of model performance 
def evaluate_model(model, dataloader, loss_fn, device="cpu"):
    """
    Evaluate the performance of the model.

    Parameters
    ----------
    model : CNN model for prediction
    dataloader : DataLoader object for evaluation
    loss_fn : Loss function
    device : 'cpu' (default) or 'cuda'

    Returns
    ----------
    Loss: Average loss
    Accuracy: Model accuracy
    """
    try:
        # Initialize total loss and correct predictions
        total_loss = 0
        total_correct = 0

        # Set model to evaluation mode
        model.eval()

        # Do not compute gradients
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloader, desc="Scoring", leave=False):
                try:
                    # Load inputs and labels to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Calculate loss
                    loss = loss_fn(outputs, labels)
                    total_loss += loss.data.item() * inputs.size(0)

                    # Compute correct predictions
                    is_correct = torch.eq(torch.argmax(outputs, dim=1), labels)
                    total_correct += torch.sum(is_correct).item()

                except Exception as e:
                    logger.info(f"Error processing batch: {str(e)}")

        return (total_loss / len(dataloader.dataset),
                total_correct / len(dataloader.dataset))

    except Exception as e:
        logger.info(f"Error in evaluate_model: {str(e)}")
        return None, None


# Do a prediction with the model
def predict(model, dataloader, device='cpu'):
    """
    Generates class prediction probabilities for each sample in the dataloader.

    Parameters
    ----------
    model : CNN model for inference
    dataloader : DataLoader object for prediction
    device : 'cpu' (default) or 'cuda'

    Returns
    ----------
    prediction_probas : Tensor containing probabilities for each class per sample
    """
    try:
        prediction_probas = torch.tensor([]).to(device)

        # No gradients needed
        with torch.inference_mode():
            for inputs, labels in tqdm(dataloader, desc="Predicting", leave=False):
                try:
                    # Load inputs to device
                    inputs = inputs.to(device)

                    # Make predictions (outputs are logits)
                    outputs = model(inputs)

                    # Convert logits to probabilities
                    proba = F.softmax(outputs, dim=1)
                    prediction_probas = torch.cat((prediction_probas, proba), dim=0)

                except Exception as e:
                    logger.info(f"Error processing batch: {str(e)}")

        return prediction_probas

    except Exception as e:
        logger.info(f"Error in predict function: {str(e)}")
        return None

# Calculate model performance metrics
def model_metrics(true_labels, pred_labels, classes):
    """Calculates accuracy, recall, precision, F1-score, and plots the confusion matrix.

    Parameters
    ----------
    true_labels : Tensor of actual labels
    pred_labels : Tensor of predicted labels
    classes : List of class names for confusion matrix display
    """
    try:
        # Calculate accuracy, recall, precision and F1
        accuracy = accuracy_score(true_labels.cpu(), pred_labels.cpu())
        recall = recall_score(true_labels.cpu(), pred_labels.cpu(),
                              average='macro')
        precision = precision_score(true_labels.cpu(), pred_labels.cpu(),
                                    average='macro', zero_division=0.0)
        f1 = f1_score(true_labels.cpu(), pred_labels.cpu(), average='macro')

        logger.info(f'Accuracy: {accuracy:.2%}')
        logger.info(f'Recall Score: {recall:.2%}')
        logger.info(f'Precision Score: {precision:.2%}')
        logger.info(f'F1 Score: {f1:.2%}')

    except Exception as e:
        logger.info(f"Error calculating performance metrics: {str(e)}")
        return  None

    try:
        # Plot the confusion matrix
        cm = confusion_matrix(true_labels.cpu(), pred_labels.cpu())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
        plt.show()

    except Exception as e:
        logger.info(f"Error plotting confusion matrix: {str(e)}")


# Callbacks
class Callbacks:
    """
    Class object for implementing callbacks
    """

    def __init__(self, save_path, patience, optimizer, step_size=4,
                 scheduler_type='reducelr'):
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path = save_path
        self.patience = patience
        self.step_size = step_size
        self.scheduler_type = scheduler_type
        self.optimizer = optimizer

    def scheduler(self, scheduler_type=None):
        """Function that implements a learning rate scheduler"""
        try:
            if scheduler_type == 'steplr':
                gamma = 0.2
                scheduler = StepLR(
                    self.optimizer,
                    step_size=self.step_size,
                    gamma=gamma,
                )
            else:
                scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=self.patience-1)

            print(type(scheduler))
            return scheduler

        except Exception as e:
            logger.info(f"An error occurred with callback-scheduler: {str(e)}")
            return None  # Gracefully handle failure

    def checkpointing(self, validation_loss, best_val_loss, model, optimizer):
        """
        Implements checkpointing - save the best model based on validation loss
        """
        try:
            if validation_loss < best_val_loss:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val_loss,
                    },
                    self.save_path,
                )
                logger.info(f"Checkpoint saved with validation loss {validation_loss:.4f}")
        except Exception as e:
            logger.info(f"An error occurred during checkpointing: {str(e)}")

    def early_stopping(self, validation_loss, best_val_loss, counter):
        """
        Function that implements Early Stopping
        """
        try:
            stop = False
            if validation_loss < best_val_loss:
                counter = 0
            else:
                counter += 1

            # Check if counter exceeds patience threshold
            if counter >= self.patience:
                stop = True

            return counter, stop

        except Exception as e:
            logger.info(f"An error occurred during early stopping: {str(e)}")
            return counter, False


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    optimizer,
    epochs=20,
    device="cpu",
    callbacks=None
):
    """
    Trains the model and tracks performance on validation data.
    Parameters
    ----------
    model : CNN model to train
    train_dataloader : DataLoader object for training
    val_dataloader : DataLoader object for validation
    loss_fn : Loss function
    optimizer : Optimizer function
    epochs : Number of epochs
    device : 'cpu' (default) or 'cuda'
    callback: instantiated class of callbacks to apply

    Returns
    ----------
    Dataframe with the :
        train_loss per epoch
        train_accuracy per epoch
        validation_loss per epoch
        validation_accuracy per epoch
    
    """
    try:
        # Track progress
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        learning_rates = []

        # Create trackers for checkpointing and early stopping
        best_val_loss = float("inf")
        early_stopping_counter = 0

        logger.info("Model evaluation before start of training...")

        # Initial evaluation
        train_loss, train_accuracy = evaluate_model(model, train_dataloader,
                                                    loss_fn, device=device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        validation_loss, validation_accuracy = evaluate_model(model,
                                                              val_dataloader,
                                                              loss_fn,
                                                              device=device)
        val_losses.append(validation_loss)
        val_accuracies.append(validation_accuracy)

        for epoch in range(1, epochs + 1):
            logger.info(f"\nStarting epoch {epoch}/{epochs}")

            # Train one epoch
            _, _, _ = train_model(model, train_dataloader, 1, loss_fn,
                                  optimizer, device, plot_acc_loss=False)

            # Evaluate training results
            train_loss, train_accuracy = evaluate_model(model,
                                                        train_dataloader,
                                                        loss_fn,
                                                        device=device)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            validation_loss, validation_accuracy = evaluate_model(model,
                                                                  val_dataloader,
                                                                  loss_fn,
                                                                  device=device)
            val_losses.append(validation_loss)
            val_accuracies.append(validation_accuracy)

            logger.info(f"Epoch: {epoch}")
            logger.info(f"Training loss: {train_loss:.4f}")
            logger.info(f"Training accuracy: {train_accuracy * 100:.4f}%")
            logger.info(f"Validation loss: {validation_loss:.4f}")
            logger.info(f"Validation accuracy: {validation_accuracy * 100:.4f}%")

            if callbacks:
                logger.info("Executing callbacks...")

                # Log learning rate
                lr = optimizer.param_groups[0]["lr"]
                learning_rates.append(lr)
                callbacks.scheduler().step(validation_loss)

                # Checkpointing
                callbacks.checkpointing(validation_loss, best_val_loss,
                                        model, optimizer)

                # Early Stopping
                early_stopping_counter, stop = callbacks.early_stopping(validation_loss,
                                                                        best_val_loss,
                                                                        early_stopping_counter)
                if stop:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss

        # Store results in a DataFrame
        results_dict = {
            'train_loss': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
        results = pd.DataFrame.from_dict(results_dict)

        return results

    except Exception as e:
        logger.info(f"An error occurred during training: {str(e)}")
        return None

# Load the model to device
def model_to_device(model):
  """   Adds the model to GPU if available otherwise adds it to cpu

  Parameters
  ----------
  model: Model object to be uploaded to GPU (if available)

  """

  try:
    if torch.cuda.is_available:
      device = 'cuda'
      model.to(device)
  except Exception as e:
    logger.info(f'cuda not available {e}')
    device = 'cpu'
    model.to(device)

  logger.info(f'model is running on: {device}')


# Plots the results from tracked training 
def plot_results(df):
    """Plots the results output from the `train` helper function.

    Parameters
    ----------
    df: Result DataFrame
    """
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Plot loss
        ax[0].plot(df['train_loss'], label="Training Loss")
        ax[0].plot(df['val_losses'], label="Validation Loss")
        ax[0].set_title("Loss over epochs")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Plot accuracy
        ax[1].plot(df['train_accuracies'], label="Training Accuracies")
        ax[1].plot(df['val_accuracies'], label="Validation Accuracies")
        ax[1].set_title("Accuracy over epochs")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        plt.show()

    except KeyError as e:
        logger.info(f"KeyError: Missing column in DataFrame - {str(e)}")
    except Exception as e:
        logger.info(f"An error occurred while plotting results: {str(e)}")





def preprocess_image(image: Image.Image, image_size=(32, 32)):
    """
    Preprocess an image for model inference.

    This function resizes the image to the specified dimensions, 
    converts it to a tensor, and normalizes it using CIFAR-10-specific 
    mean and standard deviation values.

    Parameters:
    -----------
        image (PIL.Image.Image): Input image to be processed.
        image_size (tuple): Target size for resizing (default: (32, 32)).

    Returns
    -----------
        torch.Tensor: Preprocessed image tensor ready for model input.
    
    """
    try:
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")

        # Define the preprocessing transformations
        transform = transforms.Compose([
            transforms.Resize(image_size),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                 std=[0.2023, 0.1994, 0.2010])  
        ])
        
        # Apply transformations and add batch dimension
        processed_image = transform(image).unsqueeze(0)  
        return processed_image
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Error during image preprocessing: {str(e)}")




