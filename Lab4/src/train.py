import FaceRecognitionPipeline as frp
import MyTorchWrapper as mtw
import Datasets as ds
import torch
import torch.nn as nn
import numpy as np
import os



if __name__ == "__main__":
    # Set global variables
    seed_value = 42
    use_gpu = True
    iomanager = mtw.IOManager(storage_dir="models")
    batch_size = 512
    DATASET_BASE_PATH = "data/datasets/CelebA"
    RESULTS_PATH = f"assets"


    ###########################################################################
    # Train
    ###########################################################################
    # Torch device
    device = mtw.get_torch_device(use_gpu=True, debug=True)

    # Load the dataset
    ids_file = DATASET_BASE_PATH + "/Anno/identity_CelebA_relabeled.txt"
    celeba_train = ds.CelebA(images_dir=DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/train", ids_file_path=ids_file)
    celeba_validation = ds.CelebA(images_dir=DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/test", ids_file_path=ids_file)
    train_loader = torch.utils.data.DataLoader(dataset=celeba_train, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    validation_loader = torch.utils.data.DataLoader(dataset=celeba_validation, batch_size=batch_size, pin_memory=use_gpu)

    # Training parameters
    num_epochs = 15
    learning_rate = .001
    evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss())

    # Create an instance of the model
    model = frp.superlight_network_9layers(celeba_train.num_classes, input_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    # Train the model
    trainer = mtw.Trainer(evaluation=evaluation, epochs=num_epochs, train_data_loader=validation_loader, validation_data_loader=validation_loader, io_manager=iomanager, device=device)
    train_results, validation_results = trainer.train(model, optimizer, verbose=True)


    ###########################################################################
    # Save training results
    ###########################################################################
    # Print results
    results_per_epoch = validation_results.average(num_epochs).as_dict()
    epoch_best_loss = np.argmin(results_per_epoch["loss"])
    print(f'Accuracy of the model at epoch {epoch_best_loss + 1} (epoch of lowest loss): {results_per_epoch["accuracy"][epoch_best_loss]} %')

    # Save a training summary
    summary = mtw.training_summary(model, optimizer, trainer, validation_results)
    iomanager.save_summary(summary_content=summary, model_id=trainer.model_id)

    # Compute model paramters
    print("Number of parameters of the model:", mtw.get_model_params(model))


    ###########################################################################
    # Figures
    ###########################################################################
    plotter = mtw.Plotter(trainer, train_results, validation_results)
    figures_folder = os.path.join(RESULTS_PATH, trainer.model_name)
    os.makedirs(figures_folder, exist_ok=True)
    
    # Plot loss and accuracy evolution per batch
    fig1 = plotter.plot_evaluation_per_batch(figsize=(10, 5))
    fig1.savefig(os.path.join(figures_folder, "loss_acc_per_batch.png"), dpi=500)

    # Plot train-validation loss evolution
    fig2 = plotter.plot_train_validation_loss(figsize=(10, 5))
    fig2.savefig(os.path.join(figures_folder, "train_validation_loss.png"), dpi=500)

    # Plot train-validation accuracy evolution
    fig3 = plotter.plot_train_validation_accuracy(figsize=(10, 5))
    fig3.savefig(os.path.join(figures_folder, "train_validation_accuracy.png"), dpi=500)

    # plt.show()
    