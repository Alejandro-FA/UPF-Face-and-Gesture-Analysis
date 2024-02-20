import FaceRecognitionPipeline as frp
import MyTorchWrapper as mtw
import Datasets as ds
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
import numpy as np
import os


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
    RESULTS_PATH = "assets/transfer_learning"
    DATASET_BASE_PATH = "data"
    MODEL_PATH = "model"

    if not os.path.isdir(RESULTS_PATH): # FIXME: don't do this here
        os.makedirs(RESULTS_PATH)

    ###########################################################################
    # Train
    ###########################################################################
    # Get unique ids of CelebA dataset
    num_classes = ds.get_num_unique_ids(DATASET_BASE_PATH + "/datasets/CelebA/Anno/identity_CelebA_relabeled.txt")
    
    # Load the original dataset
    original_ids_file = DATASET_BASE_PATH + "/expanded_annotations_relabeled.txt"
    original_train = ds.OriginalDataset(images_dir=DATASET_BASE_PATH + "/datasets/EXPANDED/train", ids_file_path=original_ids_file)
    original_validation = ds.OriginalDataset(images_dir=DATASET_BASE_PATH + "/datasets/EXPANDED/test", ids_file_path=original_ids_file)
    train_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=original_validation, batch_size=batch_size, pin_memory=True)
    
    # Transfer Learning (reset last fully connected layer)
    model = frp.superlight_network_9layers(num_classes=num_classes, input_channels=3) # FIXME: get num_classes from the dataset. We can't do this now because the number of labels is not consistent between our repositories
    model.load_state_dict(torch.load(MODEL_PATH + '/superlight_cnn/lab4_version/model_45-5.ckpt'))
    model.fc2 = nn.Linear(128, 80)

    # Training parameters
    num_epochs = 15
    learning_rate = .001
    evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss())

    # Train the model
    assert original_train.num_classes == original_validation.num_classes, "The number of classes in the training and validation datasets must be the same"
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    trainer = mtw.Trainer(evaluation=evaluation, epochs=num_epochs, train_data_loader=train_loader, validation_data_loader=validation_loader, io_manager=iomanager, device=device)
    model_id = trainer.model_id
    train_results, validation_results = trainer.train(model, optimizer, verbose=True)

    ###########################################################################
    # Test
    ###########################################################################
    original_test = original_validation # FIXME: create test dataset
    test_loader = validation_loader

    # Test the model with the test dataset
    tester = mtw.Tester(evaluation=evaluation, data_loader=test_loader, device=device)
    test_results = tester.test(model=model)
    test_results = test_results.average().as_dict()
    test_accuracy = test_results["accuracy"]
    print(f'Test Accuracy of the model on the {len(original_test)} test images: {test_accuracy} %')

    # Save a model summary
    summary = mtw.training_summary(model, optimizer, trainer, test_results)
    iomanager.save_summary(summary_content=summary, model_id=model_id)

    # Compute model paramters
    print("Number of parameters of the model:", mtw.get_model_params(model))


    ###########################################################################
    # Figures
    ###########################################################################
    results_per_batch = train_results.as_dict()
    initial_train_losses = results_per_batch["loss"]
    initial_train_accuracies = results_per_batch["accuracy"]
    
    # TODO: move to mtw framework
    # Plot loss and accuracy evolution with the training dataset
    fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle("Training Loss and accuracy evolution with initial conditions", fontsize=14, fontweight="bold")
    plot_acc_loss(fig2, axes, initial_train_accuracies, initial_train_losses, num_epochs, save_figure, plot_mode="both")
    if save_figure:
        plt.savefig(f"{RESULTS_PATH}/{model_id}.png", dpi=500)

    # Plot train-validation accuracy evolution
    fig3 = plt.figure(figsize=(10, 5))
    train_accuracies = train_results.average(num_epochs=num_epochs).as_dict()["accuracy"]
    validation_accuracies = validation_results.average(num_epochs=num_epochs).as_dict()["accuracy"]
    epochs = np.arange(1, num_epochs + 1)
    
    plt.plot(epochs, train_accuracies, label="Train accuracy", color="blue")
    plt.plot(epochs, validation_accuracies, label="Validation accuracy", color="red")
    plt.title("Train and validation accuracy evolution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if save_figure:
        plt.savefig(f"{RESULTS_PATH}/{model_id}_train_validation_accuracy.png", dpi=500)


    # Plot train-validation loss evolution
    fig4 = plt.figure(figsize=(10, 5))
    train_losses = train_results.average(num_epochs=num_epochs).as_dict()["loss"]
    validation_losses = validation_results.average(num_epochs=num_epochs).as_dict()["loss"]
    epochs = np.arange(1, num_epochs + 1)
    
    plt.plot(epochs, train_losses, label="Train loss", color="blue")
    plt.plot(epochs, validation_losses, label="Validation loss", color="red")
    plt.title("Train and validation loss evolution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if save_figure:
        plt.savefig(f"{RESULTS_PATH}/{model_id}_train_validation_loss.png", dpi=500)


    # plt.show()
    