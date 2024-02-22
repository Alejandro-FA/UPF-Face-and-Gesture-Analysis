import FaceRecognitionPipeline as frp
import MyTorchWrapper as mtw
import Datasets as ds
import torch
import torch.nn as nn
import numpy as np



if __name__ == "__main__":
    # Set global variables
    seed_value = None
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
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Train the model
    trainer = mtw.Trainer(evaluation=evaluation, epochs=num_epochs, train_data_loader=validation_loader, validation_data_loader=validation_loader, io_manager=iomanager, device=device)
    train_results, validation_results = trainer.train(model, optimizer, lr_scheduler, seed_value, verbose=True)


    ###########################################################################
    # Save training results
    ###########################################################################
    # Print results
    epoch_best_loss = np.argmin(validation_results["loss"])
    print(f'Accuracy of the model at epoch {epoch_best_loss + 1} (epoch of lowest loss): {validation_results["accuracy"][epoch_best_loss]} %')

    # Save a training summary
    summary = mtw.training_summary(model, optimizer, trainer, validation_results)
    iomanager.save_summary(summary_content=summary, model_id=trainer.model_id)

    # Compute model paramters
    print("Number of parameters of the model:", mtw.get_model_params(model))
    