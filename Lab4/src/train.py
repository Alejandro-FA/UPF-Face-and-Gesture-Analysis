import FaceRecognitionPipeline as frp
import MyTorchWrapper as mtw
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
import numpy as np


def plot_acc_loss(fig, axes, accuracies, losses, epochs, save_figure, plot_mode="both"):
    if plot_mode != "both" and plot_mode != "smooth" and plot_mode != "real":
        print("plot_mode options: both, smooth or both")
        return
    
    window = 100
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
    RESULTS_PATH = "assets"
    DATASET_BASE_PATH = "data/datasets/CelebA"

    ###############################################################################
    # Train
    ###############################################################################
    # Load the dataset
    celeba_train = frp.CelebA(path=DATASET_BASE_PATH + "/Img/img_align_celeba_train", ids_file=DATASET_BASE_PATH + "/Anno/identity_CelebA_train.txt")
    celeba_validation = frp.CelebA(path=DATASET_BASE_PATH + "/Img/img_align_celeba_test", ids_file=DATASET_BASE_PATH + "/Anno/identity_CelebA_test.txt") # FIXME: Rename this to validation
    train_loader = torch.utils.data.DataLoader(dataset=celeba_train, batch_size=256, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=celeba_validation, batch_size=256, pin_memory=True)

    # Training parameters
    num_epochs = 5
    learning_rate = .001
    evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss())

    # Create an instance of the model
    num_classes_train = len(celeba_train.get_unique_labels())
    num_classes_validation = len(celeba_validation.get_unique_labels())
    assert num_classes_train == num_classes_validation, "The number of classes in the training and validation datasets must be the same"
    model = frp.network_9layers(num_classes=num_classes_train, input_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    # Train the model
    model_id = iomanager.next_id_available()
    print(f"Training model {model_id} with {len(celeba_train)} images...")

    trainer = mtw.Trainer(evaluation=evaluation, epochs=num_epochs, data_loader=train_loader, device=device)
    train_results = trainer.train(model, optimizer, verbose=True)
    initial_train_losses = train_results['loss']
    initial_train_accuracies = train_results['accuracy']


    # Plot loss and accuracy evolution with the training dataset
    fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle("Training Loss and accuracy evolution with initial conditions", fontsize=14, fontweight="bold")
    plot_acc_loss(fig2, axes, initial_train_accuracies, initial_train_losses, num_epochs, save_figure, plot_mode="both")
    if save_figure:
        plt.savefig(f"{RESULTS_PATH}/fig2.png", dpi=500)
    plt.show()

    # Save the model checkpoint
    iomanager.save(model=model, model_id=model_id)


    ###############################################################################
    # Test
    ###############################################################################
    celeba_test = frp.CelebA(path="/data/datasets/CelebA/Img/img_align_celeba_test") # FIXME: create test dataset
    test_loader = torch.utils.data.DataLoader(dataset=celeba_test, batch_size=256, pin_memory=True)

    # Test the model with the test dataset
    tester = mtw.Tester(evaluation=evaluation, data_loader=test_loader, device=device)
    test_results = tester.test(model=model, verbose=True)
    initial_test_acc = test_results["accuracy"]
    print(f'Test Accuracy of the model on the {len(celeba_test)} test images: {initial_test_acc} %')

    # Save a model summary
    summary = mtw.training_summary(model, optimizer, trainer, test_results)
    iomanager.save_summary(summary_content=summary, model_id=model_id)

    # Compute model paramters
    print("Number of parameters of the model:", mtw.get_model_params(model))

    