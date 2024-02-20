import FaceRecognitionPipeline as frp
import MyTorchWrapper as mtw
import Datasets as ds
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
import numpy as np
from torchvision import transforms


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
    iomanager = mtw.IOManager(storage_dir="model/")
    save_figure = True
    batch_size = 256
    RESULTS_PATH = "assets"
    DATASET_BASE_PATH = "data/datasets/CelebA"

    ###########################################################################
    # Train
    ###########################################################################
    # Transformations
    transform = transforms.ToTensor() # Uncomment this for LightCNN
    # transform = transforms.Compose([ # Uncomment this for SqueezeNet
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    # Load the dataset
    ids_file = DATASET_BASE_PATH + "/Anno/identity_CelebA_relabeled.txt"
    celeba_train = ds.CelebA(images_dir=DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/train", ids_file_path=ids_file, transform=transform)
    celeba_validation = ds.CelebA(images_dir=DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/test", ids_file_path=ids_file, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=celeba_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=celeba_validation, batch_size=batch_size, pin_memory=True)

    # Training parameters
    num_epochs = 30
    learning_rate = .001
    evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss())

    # Create an instance of the model
    num_classes_train = celeba_train.num_classes
    num_classes_validation = celeba_validation.num_classes
    assert num_classes_train == num_classes_validation, "The number of classes in the training and validation datasets must be the same"
    # model = frp.network_9layers(num_classes=num_classes_train, input_channels=3)
    model = frp.superlight_network_9layers(num_classes=num_classes_train, input_channels=3)
    # model = frp.SqueezeNet(num_classes=num_classes_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    # Train the model
    trainer = mtw.Trainer(evaluation=evaluation, epochs=num_epochs, train_data_loader=train_loader, validation_data_loader=validation_loader, io_manager=iomanager, device=device)
    model_id = trainer.model_id
    train_results, validation_results = trainer.train(model, optimizer, verbose=True)


    ###########################################################################
    # Test
    ###########################################################################
    celeba_test = ds.CelebA(images_dir=DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/test", ids_file_path=ids_file, transform=transform) # FIXME: create test dataset
    test_loader = torch.utils.data.DataLoader(dataset=celeba_test, batch_size=batch_size, pin_memory=True)

    # Test the model with the test dataset
    tester = mtw.Tester(evaluation=evaluation, data_loader=test_loader, device=device)
    test_results = tester.test(model=model)
    test_results = test_results.average().as_dict()
    initial_test_acc = test_results["accuracy"]
    print(f'Test Accuracy of the model on the {len(celeba_test)} test images: {initial_test_acc} %')

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
    