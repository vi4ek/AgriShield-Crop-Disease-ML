import torch
import torch.nn.functional as F
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet-B0.
    Hooks into the final convolutional layer to extract feature maps and gradients,
    producing a visual attention heatmap overlay on the leaf image.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Default target layer for EfficientNet-B0 is the last conv layer in features
        if target_layer is None:
            self.target_layer = self.model.backbone.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM heatmap tensor for the input image and target class.
        """
        # Ensure tensor requires grad for backward pass
        input_tensor = input_tensor.requires_grad_(True)
        
        disease_logits, _ = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(disease_logits, dim=1).item()

        score = disease_logits[0, target_class]
        self.model.zero_grad()
        score.backward()

        # Compute weights from gradients via global average pooling
        gradients = self.gradients[0]  # shape: (C, H, W)
        activations = self.activations[0]  # shape: (C, H, W)

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)

        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        if np.max(cam) > 0:
            cam = cam / np.max(cam)  # Normalize between 0 and 1
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, original_pil_image, cam_map, alpha=0.5):
        """
        Blends Jet colormap heatmap onto the original PIL leaf image.
        Returns a PIL Image object of the heatmap overlay.
        """
        img_np = np.array(original_pil_image.convert('RGB'))
        height, width, _ = img_np.shape

        if cv2 is not None:
            # Resize heatmap to match original image dimensions
            cam_resized = cv2.resize(cam_map, (width, height))
            cam_uint8 = np.uint8(255 * cam_resized)

            # Apply Jet colormap
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Superimpose heatmap onto original image
            blended = cv2.addWeighted(img_np, 1 - alpha, heatmap_rgb, alpha, 0)
            return Image.fromarray(blended)
        else:
            # Fallback using PIL and NumPy (Jet Colormap)
            cam_pil = Image.fromarray((cam_map * 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
            cam_norm = np.array(cam_pil).astype(np.float32) / 255.0

            r = np.clip(1.5 - np.abs(4 * cam_norm - 3), 0, 1)
            g = np.clip(1.5 - np.abs(4 * cam_norm - 2), 0, 1)
            b = np.clip(1.5 - np.abs(4 * cam_norm - 1), 0, 1)
            heatmap_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

            blended = (img_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
            return Image.fromarray(blended)
