"""RNN models for body-state estimation via path integration.

Tasks:
1. Define RNN architecture (vanilla RNN, LSTM, or GRU)
2. Implement place cell output layer
3. Define training objective (cross-entropy or other)
4. Provide method to extract embeddings for Team RL
"""

from typing import Literal, Optional

import lightning as L
import torch
import torch.nn as nn


class RNN(nn.Module):
    """Vanilla RNN for state estimation.

    Takes angular velocities as input and outputs place cell activations.
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 256,
        output_size: int = 256,
        activation: Literal["tanh", "relu"] = "tanh",
    ):
        """Initialize the RNN.

        Args:
            input_size: Dimension of input (angular velocities).
            hidden_size: Number of hidden units.
            output_size: Number of place cells (output dimension).
            activation: Activation function.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: Define RNN layers
        # Hint: You can use nn.RNN or implement manually with nn.Linear
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity=activation,
        )

        # Output projection to place cell activations
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).
            hidden: Optional initial hidden state.

        Returns:
            Tuple of (place_cell_logits, hidden_states).
            place_cell_logits: Shape (batch, seq_len, output_size).
            hidden_states: Shape (batch, seq_len, hidden_size).
        """
        rnn_out, _ = self.rnn(x, hidden)
        output = self.output_proj(rnn_out)
        return output, rnn_out


class LSTM(nn.Module):
    """LSTM for state estimation."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 256,
        output_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement LSTM architecture
        raise NotImplementedError("LSTM not yet implemented")

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("LSTM not yet implemented")


class GRU(nn.Module):
    """GRU for state estimation."""

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 256,
        output_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # TODO: Implement GRU architecture
        raise NotImplementedError("GRU not yet implemented")

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("GRU not yet implemented")


class StateEstimationModel(L.LightningModule):
    """Lightning wrapper for state estimation models.

    Handles training, validation, and provides interface methods
    for other teams (get_embedding, get_hidden_states).
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 256,
        output_size: int = 256,
        model_type: Literal["rnn", "lstm", "gru"] = "rnn",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        """Initialize the Lightning module.

        Args:
            input_size: Dimension of input (angular velocities).
            hidden_size: Number of hidden units.
            output_size: Number of place cells.
            model_type: Which architecture to use.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Instantiate the appropriate model
        if model_type == "rnn":
            self.model = RNN(input_size, hidden_size, output_size)
        elif model_type == "lstm":
            self.model = LSTM(input_size, hidden_size, output_size)
        elif model_type == "gru":
            self.model = GRU(input_size, hidden_size, output_size)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # TODO: Consider alternative loss functions
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x, hidden)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute a training step."""
        velocities, targets = batch
        logits, _ = self(velocities)

        # Reshape for cross-entropy
        logits_flat = logits.view(-1, self.output_size)
        targets_flat = targets.view(-1, self.output_size)

        loss = self.loss_fn(logits_flat, targets_flat.argmax(dim=-1))

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute a validation step."""
        velocities, targets = batch
        logits, _ = self(velocities)

        logits_flat = logits.view(-1, self.output_size)
        targets_flat = targets.view(-1, self.output_size)

        loss = self.loss_fn(logits_flat, targets_flat.argmax(dim=-1))

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # =========================================================================
    # Interface methods for other teams
    # =========================================================================

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for RL (Team RL interface).

        Returns the final hidden state as the embedding.

        Args:
            x: Input trajectory of shape (batch, seq_len, input_size).

        Returns:
            Embedding of shape (batch, hidden_size).
        """
        _, hidden_states = self(x)
        return hidden_states[:, -1, :]

    def get_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """Get full hidden state trajectory (Team Interpretation interface).

        Args:
            x: Input trajectory of shape (batch, seq_len, input_size).

        Returns:
            Hidden states of shape (batch, seq_len, hidden_size).
        """
        _, hidden_states = self(x)
        return hidden_states

    @classmethod
    def load_for_embedding(cls, checkpoint_path: str) -> "StateEstimationModel":
        """Load a trained model for embedding extraction.

        Args:
            checkpoint_path: Path to model checkpoint.

        Returns:
            Loaded model in eval mode.
        """
        model = cls.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model
