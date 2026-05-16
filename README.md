
# Stable Bud: Desktop Text-to-Image Generator

An elegant GUI application built in Python that utilizes a pre-trained **Stable Diffusion** model to convert text descriptions into realistic, high-fidelity images. The software wraps powerful deep learning generation inside a streamlined desktop interface using CustomTkinter and runs hardware-accelerated processing via PyTorch.

---

## 📌 Project Features

- **Intuitive GUI Desktop App:** Uses `CustomTkinter` to provide a dark-mode responsive layout for seamless prompt submissions.
- **State-of-the-Art Generative AI:** Deploys the `CompVis/stable-diffusion-v1-4` pipeline to generate creative, contextual artwork.
- **Hardware Acceleration:** Fully optimized to execute processing using PyTorch `autocast` on CUDA-enabled NVIDIA GPUs for high-speed image rendering.
- **Auto-Save System:** Every successfully generated image is automatically saved locally to your workspace directory as `generatedimage.png`.

---

## ⚙️ How It Works (Behind the Scenes)

1. **User Input:** The user types a natural language description into the application entry box.
2. **Model Call:** Upon clicking the "Generate" button, the text is tokenized and processed by the `StableDiffusionPipeline` under half-precision floating-point format (`torch.float16`) to balance rendering performance and VRAM usage.
3. **Guidance Adjustment:** The pipeline uses a CFG (Classifier-Free Guidance) scale factor of `8.5` to ensure the final output strictly mirrors the prompt criteria.
4. **Canvas Update:** The resulting tensor array is translated via `PIL` into a graphical asset and displayed instantly on the interface canvas.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have a GPU environment configured with Python 3.8+ and standard NVIDIA CUDA drivers installed. 

### 2. Dependency Installation
Install the necessary deep learning, computer vision, and user interface packaging libraries:

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install diffusers transformers customtkinter pillow

```

### 3. Authentication Configuration

Because the Hugging Face hub restricts unauthenticated pipeline downloads for safety metrics, you need an access token to load the baseline model:

1. Generate an Access Token via your [Hugging Face Profile Settings](https://huggingface.co/settings/tokens).
2. Accept the model terms of service on the official repository page for `CompVis/stable-diffusion-v1-4`.
3. Create a secondary script named `authtoken.py` in your local directory containing your string token assignment:

```python
# authtoken.py
auth_token = "YOUR_HUGGING_FACE_ACCESS_TOKEN_HERE"

```

---

## 💻 Source Code Architecture

Run the application using the following execution command:

```bash
python app.py

```

### Core Code Snippet

```python
import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token

# Application UI Initialization
app = tk.Tk()
app.geometry("523x622")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Loading the Generative AI Model Pipeline
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        # Generate the visual matrix matching the typed prompt
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    # Save image and map it directly onto the layout canvas
    image.save("generatedimage.png")
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

```

---

## 🔮 Future Enhancements

* **Multi-Image Formats:** Add variable input fields to let users specify dimensions (e.g., 512x512, 768x768) and generation steps.
* **Batch Processing:** Enable generating grids of 4 images simultaneously from a single text prompt.
* **History Viewer:** Introduce a persistent sidebar to log previously generated prompts and thumbnail history during a live session.

```

```
