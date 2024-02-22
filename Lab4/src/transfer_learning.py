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
    iomanager = mtw.IOManager(storage_dir="models/transfer_learning")
    batch_size = 512
    DATASET_BASE_PATH = "data"
    PRETRAINED_MODEL_PATH = "models/superlight_cnn/lab4_version/model_45-5.ckpt"
    PRETRAINED_MODEL_IDS = "data/datasets/CelebA/Anno/identity_CelebA_relabeled.txt"


    ###########################################################################
    # Train
    ###########################################################################
    # Torch device
    device = mtw.get_torch_device(use_gpu=True, debug=True)

    # Load the dataset
    ids_file = DATASET_BASE_PATH + "/expanded_annotations_relabeled.txt"
    original_train = ds.OriginalDataset(images_dir=DATASET_BASE_PATH + "/datasets/EXPANDED/train", ids_file_path=ids_file)
    original_validation = ds.OriginalDataset(images_dir=DATASET_BASE_PATH + "/datasets/EXPANDED/test", ids_file_path=ids_file)
    train_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    validation_loader = torch.utils.data.DataLoader(dataset=original_validation, batch_size=batch_size, pin_memory=use_gpu)

    # Training parameters
    num_epochs = 15
    learning_rate = .001
    evaluation = mtw.AccuracyEvaluation(loss_criterion=nn.CrossEntropyLoss())

    # Transfer Learning (reset last fully connected layer)
    pretrained_classes = ds.get_num_unique_ids(PRETRAINED_MODEL_IDS)
    model = frp.superlight_network_9layers(num_classes=pretrained_classes, input_channels=3)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    model.fc2 = nn.Linear(128, 80)
    optimizer = torch.optim.Adam(model.fc2.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

    # Train the model
    trainer = mtw.Trainer(evaluation=evaluation, epochs=num_epochs, train_data_loader=train_loader, validation_data_loader=validation_loader, io_manager=iomanager, device=device)
    train_results, validation_results = trainer.train(model, optimizer, verbose=True)


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
    