from __future__ import annotations

import unittest

import torch

from modeling import apply_closed_form_residual


class AdherenceObjectiveTests(unittest.TestCase):
    def test_target_mass_loss_rewards_more_native_target_probability(self) -> None:
        target_mask = torch.tensor([[True, False], [True, False]])
        logits = torch.tensor([[0.0, 0.0], [2.0, 0.0]], requires_grad=True)
        stats = apply_closed_form_residual(
            logits,
            target_mask,
            target_adherence=0.99,
        )
        per_example_loss = -stats.native_mass.log()
        self.assertLess(
            float(per_example_loss[1].detach()),
            float(per_example_loss[0].detach()),
        )
        per_example_loss.mean().backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_residual_loss_already_rewards_target_mass(self) -> None:
        target_mask = torch.tensor([[True, False], [True, False]])
        logits = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        stats = apply_closed_form_residual(
            logits,
            target_mask,
            target_adherence=0.99,
        )
        per_example_loss = stats.residual_bias.square()
        self.assertLess(float(per_example_loss[1]), float(per_example_loss[0]))


if __name__ == "__main__":
    unittest.main()
