import MyTorchWrapper as mtw
import os
import numpy as np
import argparse

def parese_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluates the performance of a model")
    parser.add_argument("--model", "-m", type=str, required=True, help="Name of the model that wants to be evaluated.")
    parser.add_argument("--trl", "-t", action=argparse.BooleanOptionalAction, required=False, default=False, help="Indicates that the model that has to be evaluated comes from the transfer learning folder")
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parese_args()
    
    model_name = args.model
    transfer_learning = args.trl
    
    try:
        MODEL_ID = int(model_name)
    except:
        MODEL_ID = model_name
    
    ASSETS_PATH = f"assets/"
    storage_dir = "models/"
    
    if transfer_learning:
        ASSETS_PATH += "transfer_learning/"
        storage_dir += "transfer_learning/"
        
    iomanager = mtw.IOManager(storage_dir=storage_dir)


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
    
    if type(MODEL_ID) == int:
        figures_folder = os.path.join(ASSETS_PATH, f"model_{MODEL_ID}")
    else:
        figures_folder = os.path.join(ASSETS_PATH, MODEL_ID)
    
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
