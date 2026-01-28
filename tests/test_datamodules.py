"""Tests for data modules."""

import pytest

from articulated.estimation.datamodule import EstimationDataModule


class TestEstimationDataModule:
    """Tests for EstimationDataModule."""

    def test_initialization(self):
        """Test that data module initializes with correct parameters."""
        dm = EstimationDataModule(
            batch_size=16,
            seq_length=50,
            n_trajectories_train=100,
            n_trajectories_val=20,
            n_place_cells=64,
            seed=42,
        )

        assert dm.batch_size == 16
        assert dm.seq_length == 50
        assert dm.n_trajectories_train == 100
        assert dm.n_place_cells == 64

    def test_setup_not_implemented(self):
        """Setup should raise NotImplementedError until trajectory generation is implemented."""
        dm = EstimationDataModule(
            batch_size=16,
            seq_length=50,
            n_trajectories_train=10,
            n_trajectories_val=5,
            n_place_cells=64,
            seed=42,
        )

        with pytest.raises(NotImplementedError):
            dm.setup(stage="fit")
