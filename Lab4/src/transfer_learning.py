import FaceRecognitionPipeline as frp
import MyTorchWrapper as mtw
import Datasets as ds
import torch
import torch.nn as nn
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import cv2
from torchvision import transforms



if __name__ == "__main__":
    # Set global variables
    seed_value = 42
    use_gpu = True
    iomanager = mtw.IOManager(storage_dir="models/transfer_learning")
    batch_size = 512
    DATASET_BASE_PATH = "data"
    PRETRAINED_MODEL_PATH = "models/model_7/epoch-8.ckpt"
    PRETRAINED_MODEL_IDS = "data/datasets/VGG-Face2/vgg_expanded_annotations_relabeled.txt"
    
    color_transform = None
    # color_transform = cv2.COLOR_RGB2LAB

    ###########################################################################
    # Train
    ###########################################################################
    # Dataset transformations
    color_transform = cv2.COLOR_RGB2LAB
    # color_transform = None
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4964, 0.5473, 0.5568], std=[0.1431, 0.0207, 0.0262]),
    ])

    # Torch device
    device = mtw.get_torch_device(use_gpu=True, debug=True)

    # Load the dataset
    ids_file = DATASET_BASE_PATH + "/expanded_annotations_relabeled_v2.txt"
    original_train = ds.FeatureExtractorDataset(
        images_dir=DATASET_BASE_PATH + "/datasets/EXPANDED_v2/train",
        ids_file_path=ids_file,
        color_transform=color_transform,
        transform=transform,
    )
    original_validation = ds.FeatureExtractorDataset(
        images_dir=DATASET_BASE_PATH + "/datasets/EXPANDED_v2/test",
        ids_file_path=ids_file,
        color_transform=color_transform,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    validation_loader = torch.utils.data.DataLoader(dataset=original_validation, batch_size=batch_size, pin_memory=use_gpu)

    # Training parameters
    num_epochs = 200
    learning_rate = 1e-3
    evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss())

    # Transfer Learning (reset last fully connected layer)
    pretrained_classes = ds.get_num_unique_ids(PRETRAINED_MODEL_IDS)
    pretrained_params = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
    # model = frp.superlight_network_9layers(num_classes=pretrained_classes, input_channels=3)
    model = frp.superlight_cnn_v4(num_classes=pretrained_classes, input_channels=3, instance_norm=False, dropout=0.5)
    model.load_state_dict(pretrained_params)
    model.fc2 = nn.Linear(133, 80)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler_epoch = lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5, threshold=0.01, min_lr=1e-6)
    lr_scheduler_minibatch = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_epochs * len(train_loader))

    # Train the model
    trainer = mtw.Trainer(
        evaluation=evaluation,
        epochs=num_epochs,
        train_data_loader=train_loader,
        validation_data_loader=validation_loader,
        io_manager=iomanager,
        device=device,
    )
    train_results, validation_results = trainer.train(
        model=model,
        optimizer=optimizer,
        lr_scheduler_epoch=lr_scheduler_epoch,
        lr_scheduler_minibatch=None,
        verbose=True,
    )


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
    