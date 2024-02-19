import FaceRecognitionPipeline as frp
import MyTorchWrapper as mtw
import Datasets as ds
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
import numpy as np


def plot_acc_loss(fig, axes, accuracies, losses, epochs, save_figure, plot_mode="both"):
    if plot_mode != "both" and plot_mode != "smooth" and plot_mode != "real":
        print("plot_mode options: both, smooth or both")
        return
    
    window = min(100, len(accuracies))
    # smoothed_accuracies = np.convolve(train_accuracies, np.ones(window)[::-1], mode="same")
    
    epoch_range = np.arange(epochs + 1)
    
    if plot_mode != "smooth":
        axes[0].plot(accuracies, label="Real accuracies", color="#68CDFF")
    
    if plot_mode != "real":
        smoothed_accuracies = scipy.signal.savgol_filter(accuracies, window, 5)
        axes[0].plot(smoothed_accuracies, label="Smoothed accuracies", color="blue")
    
    axes[0].set_xlim(None)
    # axes[0].set_xticks(epoch_range * len(losses), epoch_range)
    axes[0].set_title("Train accuracy at each step")
    axes[0].set_xlabel("Steps (number of forward passes)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim((0, 100))
    axes[0].legend()
    axes[0].grid()

    if plot_mode != "smooth":
        axes[1].plot(losses, label="Real losses", color="#68CDFF")
    
    if plot_mode != "real":
        smoothed_losses = scipy.signal.savgol_filter(losses, window, 5)
        axes[1].plot(smoothed_losses, label="Smoothed losses", color="blue")
    
    axes[1].set_title("Loss at each step")
    axes[1].set_xlabel("Steps (number of forward passes)")
    axes[1].set_ylabel("Loss")
    axes[1].set_ylim((0, None))
    axes[1].legend()
    axes[1].grid()

    return fig



if __name__ == "__main__":
    # Set global variables
    seed_value = 42
    device = mtw.get_torch_device(use_gpu=True, debug=True)
    iomanager = mtw.IOManager(storage_dir="model/transfer_learning")
    save_figure = True
    batch_size = 256
    RESULTS_PATH = "assets"
    DATASET_BASE_PATH = "data"

    ###########################################################################
    # Train
    ###########################################################################
    # Load the dataset
    ids_file = DATASET_BASE_PATH + "/identity_CelebA_relabeled.txt"
    celeba_train = ds.CelebA(images_dir=DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/train", ids_file_path=ids_file)
    celeba_validation = ds.CelebA(images_dir=DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/test", ids_file_path=ids_file)
    train_loader = torch.utils.data.DataLoader(dataset=celeba_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=celeba_validation, batch_size=batch_size, pin_memory=True)

    # Training parameters
    num_epochs = 30
    learning_rate = .001
    evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss())

    # Create an instance of the model
    num_classes_train = celeba_train.num_unique_labels()
    num_classes_validation = celeba_validation.num_unique_labels()
    assert num_classes_train == num_classes_validation, "The number of classes in the training and validation datasets must be the same"
    model = frp.network_9layers(num_classes=num_classes_train, input_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    # Train the model
    trainer = mtw.Trainer(evaluation=evaluation, epochs=num_epochs, train_data_loader=train_loader, validation_data_loader=validation_loader, io_manager=iomanager, device=device)
    model_id = trainer.model_id
    train_results, validation_results = trainer.train(model, optimizer, verbose=True)


    