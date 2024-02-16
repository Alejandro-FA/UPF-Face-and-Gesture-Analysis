import FaceRecognitionPipeline as frp
import torch

seed_value = 42
dataset = frp.CelebA()
model = frp.network_9layers()
device = frp.get_torch_device(use_gpu=True, debug=True)

train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, pin_memory=True)


if seed_value is not None: torch.manual_seed(seed_value) # Ensure repeatable results
model.train() # Set the model in training mode
model.to(device)

total_steps = len(dataset)
feedback_step = round(total_steps / 3) + 1
results = self.evaluation.create_results()

for epoch in range(self.epochs):
    # Iterate over all batches of the dataset
    for i, (features, labels) in enumerate(self.data_loader):
        # Move the data to the torch device
        features = features.to(self.device)
        labels = labels.to(self.device) # FIXME: Perhaps we need to use .to(self.device, dtype=torch.long)

        outputs = model(features)  # Forward pass
        loss = self.evaluation(outputs, labels, results)  # Evaluation

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and ((i + 1) % feedback_step == 0 or i + 1 == total_steps):
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, self.epochs, i + 1, total_steps, loss.item()
                )
            )

# return results.as_dict(averaged=False)