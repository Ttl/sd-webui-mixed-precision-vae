# sd-webui-mixed-precision-vae

This is an extension for [AUTOMATIC111
stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

Some Stable Diffusion VAEs generate nans when using half precision format. These
NANs are caused by values exceeding the half precision floating point range of
-65504 to +65504. This extension uses single precision for layers that can
generate these high values and half precision for the rest of the layers.  When
run in mixed precision VAE uses less VRAM and is faster. The difference in the
resulting images compared to single precision VAE should be very small. This
extension does not help with nans produced in the U-Net.

# Benchmarks

SDXL model, 1024x1024 image.

Runtime:
no-half-vae: 1.8s
mixed precision: 1.6s

VAE parameter size:
fp16: 0.17 GB
mixed: 0.21 GB
fp32: 0.33 GB

Peak VRAM usage (MB):
no-half-vae: 6814
mixed precision: 6552

# Installation

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter URL of this repo to "URL for extension's git repository".
4. Press "Install" button.
5. Restart Web UI.

# Usage

There are no configurable options. The extension is applied automatically when
VAE is loaded in half precision at the start of image generation. If single
precision VAE is used (`--no-half-vae`) extension is not applied. A message is
printed to the output terminal after generating the first image if the
extensions has been applied or not.

This extensions should be compatible with Stable Diffusion versions, 1.x, 2.x
and SDXL.

