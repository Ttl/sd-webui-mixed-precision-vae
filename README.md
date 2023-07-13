# sd-webui-fp16vae

This is an extension for [AUTOMATIC111
stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

Some Stable Diffusion VAEs generate nans when using half precision format. These
NANs are caused by values exceeding the half precision floating point range of
-65504 to +65504. This extension clips the VAE decoder inputs at each layer to
avoid NAN values. The difference to single precision VAE should be
imperceptible. This extension does not help with nans produced in the U-Net.

# Installation

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter URL of this repo to "URL for extension's git repository".
4. Press "Install" button.
5. Restart Web UI.

# Usage

There are no configurable options. The clipping is applied automatically when
VAE is loaded in half precision. If single precision VAE is used
(`--no-half-vae`) no clipping is applied. When clipping is applied a message
`fp16 VAE clipping applied` is printed to terminal.

This extensions should be compatible with Stable Diffusion versions, 1.x, 2.x
and SDXL.

