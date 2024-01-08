import torch
from tqdm import tqdm
import utilties

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 1
MODEL_PATH = "/home/halas//models"
RESULT_FILE_PATH = "/home/halas/result.txt"


def train_fn(loader, model, optimizer, loss_fn):
    """Train CNN model

    Args:
        loader (torch.loader): training loader
        model (torch.model): chosen CNN model
        optimizer (torch.optimizer): chosen optimizer
        loss_fn (torch.loss_fn): chosen loss function
        scaler (torch.scaler): chosen scaler
    """
    loop = tqdm(loader)

    for _, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast(enabled=False):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def traning_cnn(model, optimizer, loss_fn, train_loader, val_loader, test_loader, model_name):

    utilties.check_accuracy_on_cnn_models(val_loader, model, RESULT_FILE_PATH, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        print(f"EPOCH NUMER {epoch}")
        train_fn(train_loader, model, optimizer, loss_fn)
        utilties.check_accuracy_on_cnn_models(val_loader, model, device=DEVICE)
        torch.save(model.state_dict(), MODEL_PATH + "EPOCH_" + model_name + "_" + str(epoch))

    utilties.save_predictions_as_imgs(test_loader, model, "saved_images/" + model_name)
