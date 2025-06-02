import numpy as np
import torch
import torch.nn.functional as F
from ogb_utils import load_dataset
import matplotlib.pyplot as plt


def maybe_augment_state(state: torch.Tensor, next_state: torch.Tensor, pixel_obs: bool, use_augs: bool):
    if pixel_obs and use_augs:
        if len(state.shape) != 5: state = state.unsqueeze(1)
        batch_size, horizon, history, height, width = state.shape

        # Group states before augmenting.
        both_state = torch.concatenate([state.reshape(-1, history, height, width), next_state.reshape(-1, history, height, width)], 0)
        both_state = shift_aug(both_state)

        state, next_state = torch.chunk(both_state, 2, 0)
        state = state.reshape(batch_size, horizon, history, height, width)
        next_state = next_state.reshape(batch_size, horizon, history, height, width)

        if horizon == 1:
            state = state.squeeze(1)
            next_state = next_state.squeeze(1)
    return state, next_state

def shift_aug(image: torch.Tensor, pad: int=4):
    batch_size, _, height, width = image.size()
    image = F.pad(image, (pad, pad, pad, pad), 'replicate')
    eps = 1.0 / (height + 2 * pad)

    arange = torch.linspace(-1.0 + eps, 1.0 - eps, height + 2 * pad, device=image.device, dtype=torch.float)[:height]
    arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)

    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    shift = torch.randint(0, 2 * pad + 1, size=(batch_size, 1, 1, 2), device=image.device, dtype=torch.float)
    shift *= 2.0 / (height + 2 * pad)
    return F.grid_sample(image, base_grid + shift, padding_mode='zeros', align_corners=False)

# Load image from .npz file
dataset = load_dataset('F:/workspace/sai/data/visual-antmaze-medium-navigate-v0-val.npz', 
                        np.uint8, compact_dataset=True)

# Convert to PyTorch tensor and add batch dimension
s = dataset['observations'][1000:1030].unsqueeze(0).type(torch.float)
ns = dataset['observations'][1001:1031].unsqueeze(0).type(torch.float)
g = dataset['observations'][1600].unsqueeze(0).type(torch.float)
print(s.shape) # 3x64x64

# augmented = shift_aug(image_tensor, pad=4)
# aug_s, aug_ns, aug_g = maybe_augment_state(s, ns, g, pixel_obs=True, use_augs=True)
# print("Original shape:", s.shape)
# print("Augmented shape:", aug_s.shape)  # Should match (1, 3, 64, 64)

# # Convert tensors to numpy for visualization
# s_img = aug_s.squeeze().permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)  # (64, 64, 3)
# ns_img = aug_ns.squeeze().permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)  # (64, 64, 3)
# g_img = aug_g.squeeze().permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)

# # Plot
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
# ax1.imshow(s_img)
# ax1.set_title('state')
# ax2.imshow(ns_img)
# ax2.set_title('next_state')
# ax3.imshow(g_img)
# ax3.set_title('goal')
# plt.show()

# augmented = shift_aug(image_tensor, pad=4)
aug_s, aug_ns = maybe_augment_state(s.clone(), ns.clone(), pixel_obs=True, use_augs=True)
print("Original shape:", s.shape)
print("Augmented shape:", aug_s.shape)  # Should match (1, 3, 64, 64)

# Convert tensors to numpy for visualization


# Plot
for i in range(29):
    
    s_img = s[0, i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)  # (64, 64, 3)
    s_aug = aug_s[0, i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)  # (64, 64, 3)
    ns_img = ns[0, i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)  # (64, 64, 3)
    ns_aug = aug_ns[0, i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)  # (64, 64, 3)

    fig, axs= plt.subplots(2, 2, figsize=(10, 10))

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

    ax1.imshow(s_img)
    ax1.set_title('original_state')
    ax2.imshow(s_aug)
    ax2.set_title('state')
    ax3.imshow(ns_img)
    ax3.set_title('original_ns')
    ax4.imshow(ns_aug)
    ax4.set_title('next_state')

    plt.show(block=False)   # Non-blocking show
    plt.pause(2)            # Wait for 1 second
    plt.close(fig)          # Close the specific figure
    