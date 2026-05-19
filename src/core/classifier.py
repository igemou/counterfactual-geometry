from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, projection_dim: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        if projection_dim is None:
            self.linear = nn.Linear(input_dim, num_classes)
            self.projection = None
            self.head = None
        else:
            self.linear = None
            self.projection = nn.Linear(input_dim, projection_dim)
            self.head = nn.Linear(projection_dim, num_classes)

    @property
    def encoded_dim(self) -> int:
        return self.projection_dim if self.projection_dim is not None else self.input_dim

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            return embeddings
        return self.projection(embeddings)

    def classify_encoded(self, encoded_embeddings: torch.Tensor) -> torch.Tensor:
        if self.head is not None:
            return self.head(encoded_embeddings)
        if self.linear is None:
            raise ValueError("Linear classifier is missing its output layer")
        return self.linear(encoded_embeddings)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classify_encoded(self.encode(embeddings))


def build_classifier(input_dim: int, num_classes: int, projection_dim: int | None = None) -> LinearClassifier:
    return LinearClassifier(input_dim=input_dim, num_classes=num_classes, projection_dim=projection_dim)


def _classification_accuracy(classifier: nn.Module, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        logits = classifier(embeddings)
        predictions = logits.argmax(dim=1)
        return float((predictions == labels).float().mean().item())


def train_linear_probe(
    classifier: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    val_embeddings: torch.Tensor | None = None,
    val_labels: torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    early_stopping_patience: int | None = 10,
    log_every: int = 1,
) -> tuple[nn.Module, dict[str, float | int | str]]:
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(embeddings, labels), batch_size=batch_size, shuffle=True)

    selection_metric = "val_accuracy" if val_embeddings is not None and val_labels is not None else "train_accuracy"
    best_score = float("-inf")
    best_epoch = 0
    best_state_dict = deepcopy(classifier.state_dict())
    epochs_without_improvement = 0

    classifier.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_examples = 0
        for batch_embeddings, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = classifier(batch_embeddings)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            batch_size_actual = batch_labels.size(0)
            epoch_loss += loss.item() * batch_size_actual
            num_examples += batch_size_actual

        classifier.eval()
        train_accuracy = _classification_accuracy(classifier, embeddings, labels)
        val_accuracy = None
        score = train_accuracy
        if val_embeddings is not None and val_labels is not None:
            val_accuracy = _classification_accuracy(classifier, val_embeddings, val_labels)
            score = val_accuracy

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state_dict = deepcopy(classifier.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if log_every > 0 and (epoch % log_every == 0 or epoch == 1 or epoch == epochs):
            log_message = (
                f"Probe epoch {epoch}/{epochs}: "
                f"train_loss={epoch_loss / max(num_examples, 1):.4f} train_acc={train_accuracy:.4f}"
            )
            if val_accuracy is not None:
                log_message += f" val_acc={val_accuracy:.4f}"
            log_message += f" best_{selection_metric}={best_score:.4f} best_epoch={best_epoch}"
            print(log_message)

        if (
            val_embeddings is not None
            and val_labels is not None
            and early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            break

        classifier.train()

    classifier.load_state_dict(best_state_dict)
    classifier.eval()
    return classifier, {
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }
