import MyTorchWrapper as mtw
import os
import numpy as np


if __name__ == "__main__":
    MODEL_ID = 1
    ASSETS_PATH = f"assets/"
    iomanager = mtw.IOManager(storage_dir="models/")


    ###########################################################################
    # Load results
    ###########################################################################
    train_results, validation_results = iomanager.load_results(MODEL_ID)

    # Print results
    epoch_best_loss = np.argmin(validation_results["loss"])
    print(f'Accuracy of the model at epoch {epoch_best_loss + 1} (epoch of lowest loss): {validation_results["accuracy"][epoch_best_loss]} %')


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

    print("Figures saved in:", figures_folder)
    # plt.show()
