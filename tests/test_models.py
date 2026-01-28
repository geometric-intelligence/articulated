"""Tests for model architectures."""

import pytest
import torch

from articulated.estimation.model import GRU, LSTM, RNN, StateEstimationModel


class TestRNN:
    """Tests for the RNN architecture."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 8
        seq_length = 50
        input_size = 6
        hidden_size = 128
        output_size = 64

        model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )

        x = torch.randn(batch_size, seq_length, input_size)
        output, hidden_states = model(x)

        assert output.shape == (batch_size, seq_length, output_size)
        assert hidden_states.shape == (batch_size, seq_length, hidden_size)


class TestLSTM:
    """Tests for LSTM architecture."""

    def test_not_implemented(self):
        """LSTM should raise NotImplementedError until implemented."""
        with pytest.raises(NotImplementedError):
            LSTM(input_size=6, hidden_size=64, output_size=32)


class TestGRU:
    """Tests for GRU architecture."""

    def test_not_implemented(self):
        """GRU should raise NotImplementedError until implemented."""
        with pytest.raises(NotImplementedError):
            GRU(input_size=6, hidden_size=64, output_size=32)


class TestStateEstimationModel:
    """Tests for the Lightning wrapper."""

    def test_forward_with_rnn(self):
        """Test forward pass with RNN backend."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
        )

        x = torch.randn(4, 20, 6)
        output, hidden_states = model(x)

        assert output.shape == (4, 20, 64)
        assert hidden_states.shape == (4, 20, 128)

    def test_get_embedding(self):
        """Test embedding extraction for RL."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
        )

        x = torch.randn(4, 20, 6)
        embedding = model.get_embedding(x)

        assert embedding.shape == (4, 128)

    def test_get_hidden_states(self):
        """Test hidden state extraction for analysis."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
        )

        x = torch.randn(4, 20, 6)
        hidden_states = model.get_hidden_states(x)

        assert hidden_states.shape == (4, 20, 128)
