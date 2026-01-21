# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Saliency-guided regularization helpers.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class FrozenSAMMaskGenerator:
    """Generate a union mask using a frozen SAM2 predictor with bbox prompts."""

    def __init__(self, sam_predictor) -> None:
        """
        Args:
            sam_predictor: A frozen SAM2 predictor instance. It should expose
                `set_image(image)` and `predict(box=..., multimask_output=False)`.
        """
        self.sam_predictor = sam_predictor

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor [3, H, W] image tensor. Use the unpadded image in RGB.
                If the image is normalized, de-normalize before passing to SAM2.
            boxes_xyxy: Tensor [N, 4] in pixel xyxy coordinates (float OK).

        Returns:
            mask: Tensor [H, W] float in {0,1} union mask.

        Notes:
            - SAM2 usage should follow:
                predictor.set_image(image_np)
                masks, scores, logits = predictor.predict(box=boxes_np, multimask_output=False)
            - Boxes come from DETR targets in normalized cxcywh; convert via:
                box_ops.box_cxcywh_to_xyxy(boxes) * [w, h, w, h].
        """
        if boxes_xyxy.numel() == 0:
            h, w = image.shape[-2:]
            return torch.zeros((h, w), device=image.device)

        if self.sam_predictor is None:
            raise RuntimeError(
                "SAM2 predictor is not initialized. Set it in main.py or pass a valid predictor."
            )

        image_np = (
            image.permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .byte()
            .cpu()
            .numpy()
        )
        boxes_np = boxes_xyxy.detach().cpu().numpy().astype("float32")

        # SAM2 API integration point: set image + predict masks with bbox prompts.
        self.sam_predictor.set_image(image_np)
        masks, _, _ = self.sam_predictor.predict(box=boxes_np, multimask_output=False)

        mask_tensor = torch.as_tensor(masks, device=image.device, dtype=torch.float32)
        return mask_tensor.any(dim=0).float()


class GradCAMForDETR:
    """Grad-CAM helper that hooks a target layer and computes saliency maps."""

    def __init__(self, target_layer: torch.nn.Module) -> None:
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        if grad_output:
            self.gradients = grad_output[0]

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()

    def compute_cam(self, score: torch.Tensor, spatial_size: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            score: scalar tensor used to compute gradients.
            spatial_size: (H, W) of the output cam.

        Returns:
            cam: Tensor [B, H, W] normalized to [0, 1].
        """
        if self.activations is None:
            raise RuntimeError("Grad-CAM activations are missing. Run a forward pass first.")

        gradients = torch.autograd.grad(
            score,
            self.activations,
            retain_graph=True,
            create_graph=True,
        )[0]
        self.gradients = gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=spatial_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        cam_flat = cam.flatten(1)
        cam_min = cam_flat.min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam_flat.max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam
