import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.metrics import classification_report

def train(model, train_dataloader, val_dataloader, optimizer, criterion, epochs):
    """
    Trains the sentiment analysis model.

    Args:
        model: The PyTorch model.
        train_dataloader: DataLoader for the training data.
        val_dataloader: DataLoader for the validation data.
        optimizer: The optimizer to use for training.
        criterion: The loss function.
        epochs: Number of training epochs.

    Returns:
        best_model_state_dict: The state dictionary of the best performing model.
    """
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1} / {epochs}")
        model.train()
        train_loss, _ = train_epoch(model, train_dataloader, optimizer, criterion)
        valid_loss, _ = evaluate(model, val_dataloader, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"\nTraining Loss: {train_loss}")
        print(f"Validation Loss: {valid_loss}")

    return best_valid_loss

def train_epoch(model, dataloader, optimizer, criterion):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        model.zero_grad()
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        loss = criterion(outputs, b_labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss, None

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            loss = criterion(outputs, b_labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss, None
