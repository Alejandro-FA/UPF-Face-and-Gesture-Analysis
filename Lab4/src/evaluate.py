import MyTorchWrapper as mtw
import FaceRecognitionPipeline as frp
import os
import numpy as np


if __name__ == "__main__":
    MODEL_ID = 3
    ASSETS_PATH = f"assets/transfer_learning"
    iomanager = mtw.IOManager(storage_dir="models/transfer_learning")
    model = frp.superlight_network_9layers(num_classes=80, input_channels=3)


    ###########################################################################
    # Load results and model
    ###########################################################################
    train_results, validation_results = iomanager.load_results(MODEL_ID)
    iomanager.load_model(model, MODEL_ID)

    # Print results
    epoch_best_loss = np.argmin(validation_results["loss"])
    print(f'Accuracy of the model at epoch {epoch_best_loss + 1} (epoch of lowest loss): {validation_results["accuracy"][epoch_best_loss]} %')

    # Compute model paramters
    print("Number of parameters of the model:", mtw.get_model_params(model))


    ###########################################################################
    # Figures
    ###########################################################################
    plotter = mtw.Plotter(train_results, validation_results)
    figures_folder = os.path.join(ASSETS_PATH, f"model_{MODEL_ID}")
    os.makedirs(figures_folder, exist_ok=True)
    
    # Plot loss and accuracy evolution per batch
    fig1 = plotter.plot_evaluation_per_batch(figsize=(10, 5))
    fig1.savefig(os.path.join(figures_folder, "loss_acc_per_batch.png"), dpi=500)

    # Plot train-validation loss evolution
    fig2 = plotter.plot_train_validation_comparison(metric='loss', figsize=(10, 5))
    fig2.savefig(os.path.join(figures_folder, "train_validation_loss.png"), dpi=500)

    # Plot train-validation accuracy evolution
    fig3 = plotter.plot_train_validation_comparison(metric='accuracy', figsize=(10, 5))
    fig3.savefig(os.path.join(figures_folder, "train_validation_accuracy.png"), dpi=500)

    # plt.show()