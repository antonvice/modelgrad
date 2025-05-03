# Dynamic Hugging Face Model Runner (Gradio App)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a Gradio web application that dynamically generates a user interface based on the inputs and outputs required by a specified Hugging Face model. Enter a model ID from the Hugging Face Hub, and the app will attempt to determine its task, check resource requirements, and provide an appropriate interface for inference using the `transformers` library's `pipeline` abstraction.

## Features

*   **Dynamic UI:** Generates Gradio input/output components based on the identified model task (e.g., text input/output for `text-generation`, image input/text output for `image-to-text`).
*   **Hugging Face Hub Integration:** Fetches model metadata (task, size estimate) directly from the Hub.
*   **Resource Check:** Compares estimated model size with available system RAM and GPU VRAM (if detected) before loading.
*   **Confirmation Step:** Asks for user confirmation before potentially downloading large models.
*   **Pipeline-Based Inference:** Uses `transformers.pipeline` for loading and running inference, supporting many common tasks out-of-the-box.
*   **MPS Support:** Detects and attempts to use Apple Silicon (MPS) backend if available.
*   **Dockerized:** Includes a `Dockerfile` for easy containerization and deployment.

## Limitations

*   **Universality:** While aiming for broad support, creating a UI for *every* possible model and task on the Hub is extremely complex. Support is limited to tasks defined in `TASK_IO_MAP` within `app/hf_app.py`. **Contributions to expand this map are highly welcome!** (See Contributing section).
*   **Complex I/O:** Models requiring highly specific or unusual input/output structures might not render correctly or work without code modification.
*   **Model-Specific Arguments:** The app uses generic pipeline arguments. Some models require specific parameters (`guidance_scale`, custom token arguments) which are not currently exposed in the UI.
*   **Dependency Management:** While common dependencies are included, some models might require additional libraries (`timm`, `diffusers`, `scipy`, etc.) not listed in `requirements.txt`. You may need to add them manually if you encounter import errors for specific models.
*   **Resource Estimation:** Model size is an *estimate* based on common weight files. Actual memory usage can differ significantly.
*   **`gr.Examples` Update:** Uses a workaround (global variable + side-effect update) due to apparent limitations in targeting `gr.Examples` directly as an output dependency in Gradio event listeners.

## Project Structure

```
hf-dynamic-app/
├── app/
│   └── hf_app.py        # Main Python script
├── .dockerignore        # Files to ignore for Docker build context
├── .gitignore           # Files to ignore for Git
├── Dockerfile           # Docker build definition
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## Setup and Running

### Prerequisites

*   Python 3.9+
*   Git
*   Docker (for containerized running)
*   **(Optional but Recommended)** Hugging Face Account and API Token:
    *   Create an account on [huggingface.co](https://huggingface.co/).
    *   Generate an API token with at least `read` access in your account settings ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
    *   This token is needed to access gated models or private repositories.

### Local Development

1.  **Clone:** `git clone <your-repo-url>`
2.  **Navigate:** `cd hf-dynamic-app`
3.  **Create Virtual Environment:** `python -m venv .venv`
4.  **Activate Environment:** (See previous README for OS-specific commands)
5.  **Install Dependencies:** `pip install -r requirements.txt`
6.  **Set Hugging Face Token (Important!):** Set the `HF_TOKEN` environment variable.
    *   **macOS/Linux:** `export HF_TOKEN='hf_YOUR_TOKEN_HERE'`
    *   **Windows (Command Prompt):** `set HF_TOKEN=hf_YOUR_TOKEN_HERE`
    *   **Windows (PowerShell):** `$env:HF_TOKEN='hf_YOUR_TOKEN_HERE'`
    *   *(Optional: Use a `.env` file and `python-dotenv` if preferred)*
7.  **(Optional) Login via CLI:** You can also run `huggingface-cli login` which might store the token more persistently, but setting the environment variable is often sufficient for scripts.
8.  **Run:** `python -m app.hf_app` (Run as a module due to relative imports)
9.  Access the app in your browser, usually at `http://127.0.0.1:7860`.

### Docker

1.  **Prerequisites:** Docker installed and running.
2.  **Build the Image:** `docker build -t hf-dynamic-app .`
3.  **Run the Container:**

    *   **CPU Only:**
        ```bash
        # Replace hf_YOUR_TOKEN_HERE with your actual token
        docker run -it --rm -p 7860:7860 \
               -v hf_cache:/app/.cache/huggingface \
               -e HF_TOKEN='hf_YOUR_TOKEN_HERE' \
               hf-dynamic-app
        ```
    *   **With NVIDIA GPU:**
        ```bash
        # Replace hf_YOUR_TOKEN_HERE with your actual token
        docker run -it --rm --gpus all -p 7860:7860 \
               -v hf_cache:/app/.cache/huggingface \
               -e HF_TOKEN='hf_YOUR_TOKEN_HERE' \
               hf-dynamic-app
        ```
    *   **Explanation:**
        *   ... (other flags same as before) ...
        *   `-e HF_TOKEN='...'`: **Crucially passes your Hugging Face token** as an environment variable into the running container. This allows the app inside the container to authenticate with the Hub.

4.  **Access:** Open your web browser to `http://localhost:7860`.

## Configuration & Extension

*   **Adding Task Support:** Modify the `TASK_IO_MAP` dictionary within `app/task_config.py`.
*   **Dependencies:** Add to `requirements.txt` and rebuild Docker image.

## Contributing

Contributions are welcome! The primary area for expansion is the `TASK_IO_MAP` dictionary in `app/hf_app.py`.

*   **Goal:** Increase the number of Hugging Face `pipeline` tasks supported by the dynamic UI.
*   **How:** Add new entries to `TASK_IO_MAP` following the structure outlined in the "Configuration & Extension" section and demonstrated by the existing entries.
*   **Testing:** Ensure your added components, preprocessing, and postprocessing work correctly with example models for the task you are adding.
*   **Pull Requests:** Please feel free to submit a Pull Request with your enhancements or open an Issue to discuss new tasks or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file (if you choose to add one) for details.
