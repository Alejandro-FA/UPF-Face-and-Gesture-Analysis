import FaceRecognitionPipeline as frp
import MyTorchWrapper as mtw
import Datasets as ds
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import argparse
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a feature extractor model")
    parser.add_argument("--dataset", "-d", required=True, choices=["vggface2", "celeba"], help="Dataset that has to be loaded.")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    print(f"Dataset {dataset} will be used")
    
    
    # Set global variables
    seed_value = None
    use_gpu = True
    iomanager = mtw.IOManager(storage_dir="models")
    batch_size = 1024
    # color_transform = None
    color_transform = cv2.COLOR_RGB2LAB
    RESULTS_PATH = f"assets"
    CELEBA_DATASET_BASE_PATH = "data/datasets/CelebA"
    VGGFACE2_DATASET_BASE_PATH = "data/datasets/VGG-Face2"


    ###########################################################################
    # Train
    ###########################################################################
    # Torch device
    device = mtw.get_torch_device(use_gpu=True, debug=True)

    # Load the dataset
    if dataset == "celeba":
        ids_file = CELEBA_DATASET_BASE_PATH + "/Anno/identity_CelebA_relabeled.txt"
        train_dir = CELEBA_DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/train"
        test_dir = CELEBA_DATASET_BASE_PATH + "/Img/img_align_celeba_cropped/test"
    elif dataset == "vggface2":
        ids_file = VGGFACE2_DATASET_BASE_PATH + "/vgg_expanded_annotations_relabeled.txt"
        train_dir = VGGFACE2_DATASET_BASE_PATH + "/data/clean/train"
        test_dir = VGGFACE2_DATASET_BASE_PATH + "/data/clean/test"
    
    train_dataset = ds.FeatureExtractorDataset(
        images_dir=train_dir,
        ids_file_path=ids_file,
        color_transform=color_transform,
    )    
    validation_dataset = ds.FeatureExtractorDataset(
        images_dir=test_dir,
        ids_file_path=ids_file,
        color_transform=color_transform,
    )
    

    # TODO: Try Xavier initialization for convolutional layers and Gaussian for the fully connected layers.
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, pin_memory=use_gpu)

    # Training parameters
    num_epochs = 40
    learning_rate = 1e-3
    evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss())

    # Create an instance of the model
    model = frp.superlight_network_9layers(train_dataset.num_classes, input_channels=3)

    # Optimizer and a learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    lr_scheduler_epoch = lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5, threshold=0.01, min_lr=1e-5)
    lr_scheduler_minibatch = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_epochs * len(train_loader))

    # Train the model
    trainer = mtw.Trainer(
        evaluation=evaluation,
        epochs=num_epochs,
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        io_manager=iomanager,
        device=device
    )
    train_results, validation_results = trainer.train(
        model=model,
        optimizer=optimizer,
        lr_scheduler_epoch=lr_scheduler_epoch,
        lr_scheduler_minibatch=None,
        seed_value=seed_value,
        verbose=True
    )


    ###########################################################################
    # Save training results
    ###########################################################################
    # Print last learning rate used
    print("Last learning rate used:", lr_scheduler_epoch.get_last_lr())

    # Print results
    epoch_best_loss = np.argmin(validation_results["loss"])
    print(f'Accuracy of the model at epoch {epoch_best_loss + 1} (epoch of lowest loss): {validation_results["accuracy"][epoch_best_loss]} %')

    # Save a training summary
    summary = mtw.training_summary(model, optimizer, trainer, validation_results)
    iomanager.save_summary(summary_content=summary, model_id=trainer.model_id)

    # Compute model paramters
    print("Number of parameters of the model:", mtw.get_model_params(model))
    