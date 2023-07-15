# sd-webui-mixed-precision-vae

This is an extension for [AUTOMATIC111
stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

Some Stable Diffusion VAEs generate nans when using half precision format. These
nans are caused by values exceeding the half precision floating point range of
-65504 to +65504. This extension dynamically uses single precision only for
layers that exceed the half precision range. Compared to using `--no-half-vae`
that casts all weights to single precision this saves VRAM and is faster. If the
VAE does not blow up in half precision then no layers are converted avoiding
overhead. The difference in the resulting images compared to single precision
VAE should be extremely small.

# Benchmarks

SDXL model, 1024x1024 image.

VAE decoder runtime:
no-half-vae: 1.8s
mixed precision: 1.6s

VAE parameter size:
fp16: 0.167 GB
mixed (txt2img): 0.169 GB
mixed (img2img): 0.195 GB
fp32: 0.335 GB

# Installation

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter URL of this repo to "URL for extension's git repository".
4. Press "Install" button.
5. Restart Web UI.

# Usage

There are no configurable options. The extension is applied automatically at the
start of image generation if VAE has been loaded in half precision. If single
precision VAE is used (`--no-half-vae`) the extension is not applied. A message
is printed to the output terminal after generating the first image if the
extensions has been applied or not. A message is also printed if any of the
layers are converted to single precision.

This extensions should be compatible with Stable Diffusion versions, 1.x, 2.x
and SDXL.

