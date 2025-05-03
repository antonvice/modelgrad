# coding: utf-8
import gradio as gr
from huggingface_hub import HfApi, model_info, list_repo_files
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from transformers import pipeline, AutoConfig
from transformers.pipelines import PIPELINE_REGISTRY
import psutil
import torch
import humanize
import logging
import traceback
import gc # For garbage collection, especially after OOM
import re # For regex used in helper functions
import sys # To check platform
from task_config import TASK_IO_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
MAX_COMPONENTS = 5 # Max number of input/output components expected for any task

# --- Helper Functions ---

def get_compute_device_info():
    """Detects available compute device (CUDA, MPS, CPU) and returns info."""
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
            mem_str = humanize.naturalsize(total_mem_bytes, binary=True)
            return {"type": "CUDA", "id": 0, "name": device_name, "memory": mem_str}
        except Exception as e:
            logger.warning(f"Could not get CUDA device properties: {e}")
            return {"type": "CUDA", "id": 0, "name": "NVIDIA GPU (Unknown)", "memory": "N/A"}
    # Check for MPS (Apple Silicon GPU)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
         return {"type": "MPS", "id": "mps", "name": "Apple Silicon GPU", "memory": "N/A (Unified Memory)"}
    else:
        return {"type": "CPU", "id": -1, "name": "CPU", "memory": "N/A"}

def get_ram_info():
    """Returns available and total system RAM."""
    mem = psutil.virtual_memory()
    return {
        "available_str": humanize.naturalsize(mem.available, binary=True),
        "total_str": humanize.naturalsize(mem.total, binary=True),
        "available_bytes": mem.available
    }

def get_model_size_estimate_bytes(model_id: str) -> int:
    """Estimates model size in bytes by summing relevant weight files."""
    try:
        try:
            logger.info(f"Attempting to fetch model info with file metadata for {model_id}")
            model_meta = api.model_info(model_id, files_metadata=True)
            size_lookup = {sibling.rfilename: sibling.size for sibling in model_meta.siblings if sibling.size is not None}
            if not size_lookup: raise ValueError("Empty file metadata")
            logger.info(f"Using file metadata for size estimation.")
            relevant_files = [f for f in size_lookup.keys() if f.endswith(('.bin', '.safetensors'))]
        except (ValueError, TypeError, Exception) as meta_e:
            logger.warning(f"Could not use file metadata ({meta_e}), using list_repo_files fallback for {model_id}")
            files = list_repo_files(model_id, repo_type="model")
            relevant_files = [f for f in files if f.endswith(('.bin', '.safetensors'))]
            try:
                model_meta = api.model_info(model_id); size_lookup = {sibling.rfilename: sibling.size for sibling in model_meta.siblings if sibling.size is not None}
            except Exception as info_e: logger.error(f"Could not fetch model_info for {model_id} in fallback: {info_e}"); return 0
        total_size_bytes = 0; safetensors_bases = {f.replace('.safetensors', '') for f in relevant_files if f.endswith('.safetensors')}
        files_to_sum = []; processed_bases = set()
        for f in relevant_files:
            is_safetensor = f.endswith('.safetensors'); base = f
            if is_safetensor: base = base.replace('.safetensors', '')
            elif f.endswith('.bin'): base = base.replace('.bin', '')
            match = re.search(r'-\d{5}-of-\d{5}$', base);
            if match: base = base[:match.start()]
            if is_safetensor:
                if base not in processed_bases: files_to_sum.append(f); processed_bases.add(base)
            elif f.endswith('.bin') and base not in safetensors_bases and base not in processed_bases: files_to_sum.append(f); processed_bases.add(base)
        if not files_to_sum and relevant_files: logger.warning("Weight file selection logic incomplete..."); files_to_sum = relevant_files
        if not files_to_sum: logger.warning(f"Could not identify primary weight files for {model_id}."); return 0
        logger.info(f"Estimating size based on files: {files_to_sum}")
        for filename in files_to_sum:
            if filename in size_lookup: total_size_bytes += size_lookup[filename]
            else: logger.warning(f"Could not get size for listed file {filename}.")
        return total_size_bytes
    except (GatedRepoError, RepositoryNotFoundError) as repo_e: raise repo_e
    except Exception as e: logger.error(f"Error estimating model size for {model_id}: {e}", exc_info=True); return 0

def get_model_task(model_id: str):
    """Tries to determine the model's task using pipeline_tag or AutoConfig."""
    try:
        info = model_info(model_id)
        if info.pipeline_tag:
            task = info.pipeline_tag; logger.info(f"Task '{task}' found in pipeline_tag for {model_id}.")
            if task.startswith("translation"): return "translation_xx_to_yy"
            if task == "text2text-generation": logger.info("Mapping task 'text2text-generation' to UI key 'text-generation'."); return "text-generation"
            return task
        logger.warning(f"No pipeline_tag found for {model_id}. Attempting config inference.")
        config = AutoConfig.from_pretrained(model_id)
        if hasattr(config, "architectures") and config.architectures:
             arch = config.architectures[0]; logger.info(f"Trying inference based on architecture: {arch}")
             if "ForSequenceClassification" in arch: return "text-classification"
             if "ForTokenClassification" in arch and "token-classification" in TASK_IO_MAP: return "token-classification"
             if "ForQuestionAnswering" in arch: return "question-answering"
             if "ForMaskedLM" in arch: return "fill-mask"
             if "ForCausalLM" in arch: return "text-generation"
             if "ForConditionalGeneration" in arch:
                 if hasattr(config, "task_specific_params") and config.task_specific_params:
                      if "summarization" in config.task_specific_params: return "summarization"
                      if "translation" in config.task_specific_params: return "translation_xx_to_yy"
                 logger.warning("ConditionalGeneration architecture found, defaulting to 'text-generation' UI."); return "text-generation"
             if "ForImageClassification" in arch: return "image-classification"
             if "VisionEncoderDecoderModel" in arch: return "image-to-text"
             if "StableDiffusionPipeline" in arch: return "text-to-image"
        if config.model_type in PIPELINE_REGISTRY.model_mapping:
            potential_tasks = PIPELINE_REGISTRY.model_mapping[config.model_type]
            if potential_tasks:
                for task_info in potential_tasks:
                    task = task_info['task']; mapped_task = "translation_xx_to_yy" if task.startswith("translation") else task
                    if mapped_task == "text2text-generation": mapped_task = "text-generation"
                    if mapped_task in TASK_IO_MAP: logger.info(f"Inferred task '{task}' (mapped to UI key '{mapped_task}') from model_type '{config.model_type}'."); return mapped_task
                logger.warning(f"Tasks {potential_tasks} found for model_type '{config.model_type}', but none have a UI mapping.")
        logger.warning(f"Could not reliably determine task for {model_id} from config."); return None
    except (GatedRepoError, RepositoryNotFoundError) as repo_e: raise repo_e
    except Exception as e: logger.error(f"Error getting model task/config for {model_id}: {e}", exc_info=True); return None

# --- Gradio Interface Logic ---
app_state = { "pipeline": None, "task": None, "model_id": None, "io_map": None }
api = HfApi()
compute_device_info = get_compute_device_info()

def reset_app_state():
    """Safely reset the application state and clean up resources."""
    global app_state; global examples_display # Need access to potentially update examples
    pipe = app_state.get("pipeline")
    if pipe is not None:
        model_id = app_state.get('model_id', 'Unknown'); logger.info(f"Clearing pipeline object for model {model_id}")
        try:
            if hasattr(pipe, 'model') and hasattr(pipe.model, 'cpu'): pipe.model.cpu()
            del pipe; app_state["pipeline"] = None; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache(); logger.info("CUDA cache cleared.")
            elif compute_device_info['type'] == "MPS": pass # No explicit MPS cache clear
        except Exception as e: logger.warning(f"Exception during pipeline cleanup for {model_id}: {e}", exc_info=True)
    app_state = {"pipeline": None, "task": None, "model_id": None, "io_map": None}
    logger.info("App state reset.")


def check_model(model_id: str):
    """Fetches model info, estimates size, determines task, and prepares confirmation."""
    reset_app_state()
    if not model_id or "/" not in model_id or len(model_id.split('/')) != 2:
        return gr.update(visible=False), gr.update(value="Invalid Model ID format. Use 'org/model' or 'user/model'."), "", ""
    logger.info(f"Checking model: {model_id}")
    try:
        task = get_model_task(model_id); size_bytes = get_model_size_estimate_bytes(model_id)
        ram_info = get_ram_info()
        if task is None: return gr.update(visible=False), gr.update(value=f"Could not determine task for '{model_id}'."), "", ""
        ui_task_key = task
        if task == "text2text-generation": ui_task_key = "text-generation"
        elif task.startswith("translation_"): ui_task_key = "translation_xx_to_yy"
        if ui_task_key not in TASK_IO_MAP: return gr.update(visible=False), gr.update(value=f"Task '{task}' found, but no UI defined (key: '{ui_task_key}')."), "", ""
        size_str = humanize.naturalsize(size_bytes, binary=True) if size_bytes > 0 else "Unknown"
        compute_device_line = ""
        if compute_device_info['type'] == "CUDA": compute_device_line = f"*   **Detected GPU:** {compute_device_info['name']} ({compute_device_info['memory']} VRAM)"
        elif compute_device_info['type'] == "MPS": compute_device_line = f"*   **Detected Accelerator:** {compute_device_info['name']} (MPS)"
        else: compute_device_line = f"*   **Detected Accelerator:** CPU"
        details_md = f"**📋 Details:**\n*   **Model ID:** `{model_id}`\n*   **Identified Task:** `{task}`\n*   **Using UI for:** `{ui_task_key}`"
        resources_md = f"**💾 Resource Check:**\n*   **Estimated Size:** `{size_str}`\n*   **System RAM:** {ram_info['available_str']} Available / {ram_info['total_str']} Total\n{compute_device_line}"
        warnings_list = []
        if size_bytes > ram_info['available_bytes'] and size_bytes > 0: warnings_list.append("Estimated model size may exceed available system RAM.")
        if compute_device_info['type'] == "CUDA":
            try:
                 match = re.match(r"([\d\.]+)\s*([GMK])i?B", compute_device_info['memory']); val = float(match.group(1)); unit = match.group(2)
                 multiplier = {'G': 1024**3, 'M': 1024**2, 'K': 1024**1}[unit]; gpu_mem_bytes = int(val * multiplier)
                 if size_bytes > gpu_mem_bytes and size_bytes > 0: warnings_list.append("Estimated model size may exceed total GPU VRAM.")
            except Exception as parse_e: logger.warning(f"Error parsing GPU memory string: {parse_e}")
        warnings_md = "";
        if warnings_list: warnings_formatted = "\n".join([f"*   {w}" for w in warnings_list]); warnings_md = f"**⚠️ Warnings:**\n{warnings_formatted}"
        notes_md = "---\n**ℹ️ Notes:**\n*   Models download to cache (`~/.cache/huggingface/hub`).\n*   Memory usage varies (precision, inputs, etc.)."
        confirmation_md = f"### Model Confirmation\n{details_md}\n\n{resources_md}\n\n{warnings_md}\n\n{notes_md}"
        confirmation_md = "\n".join([line.strip() for line in confirmation_md.strip().splitlines()])
        return gr.update(visible=True), confirmation_md, model_id, task
    except (GatedRepoError, RepositoryNotFoundError) as repo_e:
        logger.warning(f"{type(repo_e).__name__} for model {model_id}")
        msg = f"Model `{model_id}` is gated. Request access and log in (`huggingface-cli login`)." if isinstance(repo_e, GatedRepoError) else f"Model `{model_id}` not found."
        return gr.update(visible=False), msg, "", ""
    except Exception as e:
        logger.exception(f"Error checking model {model_id}:")
        error_md = f"**An error occurred while checking the model:**\n```\n{traceback.format_exc()}\n```"
        return gr.update(visible=False), error_md, "", ""

# --- Helper Function to generate component updates (Revised for safety) ---
def _generate_component_updates(components_factory, max_components):
    """Helper to create gr.update list for placeholders based on actual components."""
    actual_components = components_factory()
    updates = []
    common_props = ["label", "interactive", "value", "visible"]
    textbox_props = ["lines", "type"]
    for i in range(max_components):
        if i < len(actual_components):
            new_comp = actual_components[i]; update_args = {}
            for prop in common_props:
                if hasattr(new_comp, prop): update_args[prop] = getattr(new_comp, prop)
            if isinstance(new_comp, gr.Textbox):
                for prop in textbox_props:
                    if hasattr(new_comp, prop):
                        prop_value = getattr(new_comp, prop)
                        if prop == 'type' and prop_value not in ["text", "password", "email"]: logger.warning(f"Invalid type '{prop_value}' for Textbox ignored."); continue
                        update_args[prop] = prop_value
            elif isinstance(new_comp, gr.Image): pass # Don't copy Image 'type' to Textbox placeholder
            update_args["value"] = None; update_args["visible"] = True
            updates.append(gr.update(**update_args))
        else: updates.append(gr.update(visible=False, value=None))
    return updates, actual_components


# --- Main function for loading model and setting up UI (Regular Function) ---
def load_model_and_setup_ui(current_model_id: str, current_task: str):
    """Loads model, updates state, configures UI. Returns tuple of updates."""
    global app_state; global examples_display # Need global examples_display for side-effect update
    if not current_model_id or not current_task:
        logger.error("load_model_and_setup_ui called with missing model_id or task.")
        # Match outputs list length for confirm_button (15 items)
        error_updates = (gr.update(visible=False), gr.update(visible=False), gr.update(value="Error: Invalid state."), *[gr.update(visible=False)] * MAX_COMPONENTS, *[gr.update(visible=False)] * MAX_COMPONENTS, gr.update())
        return tuple(error_updates)

    logger.info(f"Requesting load for model '{current_model_id}' | task '{current_task}'")
    ui_task_key = current_task
    if current_task == "text2text-generation": ui_task_key = "text-generation"
    elif current_task.startswith("translation_"): ui_task_key = "translation_xx_to_yy"
    if ui_task_key not in TASK_IO_MAP:
        error_msg = f"Error: UI not defined for task {current_task} (key: {ui_task_key})."
        logger.error(error_msg)
        # Match outputs list length (15 items)
        error_ui_updates = (gr.update(visible=False), gr.update(visible=False), gr.update(value=error_msg), *[gr.update(visible=False)] * MAX_COMPONENTS, *[gr.update(visible=False)] * MAX_COMPONENTS, gr.update())
        return tuple(error_ui_updates)

    task_config = TASK_IO_MAP[ui_task_key]; pipeline_args = task_config.get("pipeline_args", {})
    device_to_use = None; use_device_map = False
    if compute_device_info['type'] == "CUDA": use_device_map = True; device_to_use = 0
    elif compute_device_info['type'] == "MPS": device_to_use = "mps"
    else: device_to_use = -1
    load_method_log = f"{'device_map=auto' if use_device_map else f'device={device_to_use}'}"
    logger.info(f"Attempting load using: {load_method_log}")

    try:
        loaded_pipeline = None; load_error = None
        pipeline_load_task = current_task if not current_task.startswith("translation") else None
        load_attempts = [];
        if use_device_map: load_attempts.append({"type": "device_map", "args": {"device_map": "auto"}})
        if device_to_use is not None and not use_device_map: load_attempts.append({"type": f"device={device_to_use}", "args": {"device": device_to_use}})
        if device_to_use != -1: load_attempts.append({"type": "CPU fallback", "args": {"device": -1}})

        for attempt in load_attempts:
            logger.info(f"Load attempt: {attempt['type']}")
            try:
                loaded_pipeline = pipeline(task=pipeline_load_task, model=current_model_id, **attempt['args'], **pipeline_args)
                logger.info(f"Successfully loaded pipeline using {attempt['type']}."); load_error = None; break
            except (torch.cuda.OutOfMemoryError, MemoryError) as mem_e: logger.warning(f"OOM Error during {attempt['type']} load: {mem_e}. Cleaning up."); load_error = mem_e; del loaded_pipeline; loaded_pipeline = None; gc.collect(); # ... cache clear ...
            except Exception as e: logger.warning(f"Error during {attempt['type']} load: {e}", exc_info=True); load_error = e; del loaded_pipeline; loaded_pipeline = None; gc.collect()

        if loaded_pipeline is None:
            if load_error: raise load_error
            else: raise RuntimeError(f"Unknown error: Pipeline for {current_model_id} could not be loaded after all attempts.")

        app_state["pipeline"] = loaded_pipeline; app_state["task"] = current_task
        app_state["model_id"] = current_model_id; app_state["io_map"] = task_config
        input_updates, actual_input_components = _generate_component_updates(task_config["inputs"], MAX_COMPONENTS)
        output_updates, _ = _generate_component_updates(task_config["outputs"], MAX_COMPONENTS)
        example_inputs_data = task_config.get("example_inputs", [])
        formatted_examples = [];
        if example_inputs_data:
             if isinstance(example_inputs_data[0], list): formatted_examples = example_inputs_data
             else: formatted_examples = [example_inputs_data]

        # --- Side Effect Update for examples_display ---
        if examples_display: # Ensure it exists
            examples_display.examples = formatted_examples
            examples_display.inputs = actual_input_components
        else:
            logger.warning("examples_display global variable not set when trying to update.")
        # --------------------------------------------

        logger.info(f"Model {current_model_id} loaded successfully. UI setup complete.")
        # Return final tuple matching outputs list length (15 items)
        final_updates = (gr.update(visible=False), gr.update(visible=True), gr.update(value=f"**Model:** `{current_model_id}` | **Task:** `{current_task}`"), *input_updates, *output_updates, gr.update())
        return tuple(final_updates)

    except (torch.cuda.OutOfMemoryError, MemoryError) as e:
        error_type = type(e).__name__; logger.error(f"{error_type} loading {current_model_id}: {e}", exc_info=True)
        error_message = f"❌ **Error:** Ran out of memory (`{error_type}`) loading `{current_model_id}`. Try a smaller model or ensure resources."
        reset_app_state()
        if examples_display: examples_display.examples = []; examples_display.inputs = []
        # Match outputs list length (15 items)
        error_oom_updates = (gr.update(visible=False), gr.update(visible=False), gr.update(value=error_message), *[gr.update(visible=False, value=None)] * MAX_COMPONENTS, *[gr.update(visible=False, value=None)] * MAX_COMPONENTS, gr.update())
        return tuple(error_oom_updates)
    except Exception as e:
        logger.exception(f"Failed to load/setup pipeline for {current_model_id}:")
        reset_app_state()
        detailed_error = f"❌ **Error loading model:** `{current_model_id}`\n```\n{traceback.format_exc()}\n```"
        if examples_display: examples_display.examples = []; examples_display.inputs = []
        # Match outputs list length (15 items)
        error_exc_updates = (gr.update(visible=False), gr.update(visible=False), gr.update(value=detailed_error), *[gr.update(visible=False, value=None)] * MAX_COMPONENTS, *[gr.update(visible=False, value=None)] * MAX_COMPONENTS, gr.update())
        return tuple(error_exc_updates)

# --- Function Definition for input change handler (Regular Function) ---
def handle_input_change():
    """Resets state and UI elements when the model input text is changed."""
    global examples_display # Need access for side effect update
    reset_app_state()
    logger.info("Model input changed, resetting state and UI.")
    if examples_display: # Check if it exists yet
        examples_display.examples = []
        examples_display.inputs = []
    # Return tuple matching model_input.change outputs list length/order (16 items)
    return (gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(value="## Model Interface"), *[gr.update(visible=False, value=None)] * MAX_COMPONENTS, *[gr.update(visible=False, value=None)] * MAX_COMPONENTS, gr.update())

# --- Function to run inference ---
# (run_inference remains the same)
def run_inference(*inputs):
    """Runs the loaded pipeline with provided inputs after preprocessing."""
    global app_state
    pipe = app_state.get("pipeline")
    io_map = app_state.get("io_map")
    task = app_state.get("task", "unknown")
    model_id = app_state.get("model_id", "unknown")
    if pipe is None or io_map is None:
        logger.error("Run Inference called but pipeline or io_map not loaded in state.")
        error_updates = [gr.update(value="Error: Model not loaded.")] + [gr.update(value=None)] * (MAX_COMPONENTS - 1)
        return tuple(error_updates)
    try: num_expected_inputs = len(io_map["inputs"]())
    except Exception as e:
         logger.error(f"Could not determine expected inputs from io_map: {e}")
         error_updates = [gr.update(value="Error: UI configuration issue.")] + [gr.update(value=None)] * (MAX_COMPONENTS - 1)
         return tuple(error_updates)
    actual_inputs = inputs[:num_expected_inputs]
    logger.info(f"Running inference for task '{task}' on model '{model_id}' with {len(actual_inputs)} inputs.")
    preprocess_fn = io_map["preprocess"]
    postprocess_fn = io_map["postprocess"]
    num_outputs = len(io_map["outputs"]())
    try:
        logger.debug(f"Preprocessing inputs: {actual_inputs}")
        pipeline_input = preprocess_fn(*actual_inputs)
        logger.debug(f"Pipeline input: {pipeline_input}")
        with torch.no_grad():
             if isinstance(pipeline_input, dict): result = pipe(**pipeline_input)
             else: result = pipe(pipeline_input)
        logger.info(f"Raw inference result type: {type(result)}")
        logger.debug(f"Raw inference result: {result}")
        processed_result = postprocess_fn(result)
        logger.debug(f"Postprocessed result: {processed_result}")
        output_updates = []
        if num_outputs == 1: output_updates.append(gr.update(value=processed_result))
        elif isinstance(processed_result, (list, tuple)) and len(processed_result) == num_outputs:
             for res_item in processed_result: output_updates.append(gr.update(value=res_item))
        else:
             logger.warning(f"Postprocessor for task '{task}' returned {type(processed_result)}, expected tuple/list of length {num_outputs}. Adapting.")
             output_updates.append(gr.update(value=processed_result))
             for _ in range(num_outputs - 1): output_updates.append(gr.update(value=None))
        while len(output_updates) < MAX_COMPONENTS:
             output_updates.append(gr.update(visible=False, value=None))
        return tuple(output_updates)
    except Exception as e:
        logger.exception("Error during inference pipeline:")
        error_message = f"**Inference Error:**\n```\n{traceback.format_exc()}\n```"
        error_updates = [gr.update(value=error_message)]
        for i in range(1, MAX_COMPONENTS):
            is_visible = i < num_outputs
            error_updates.append(gr.update(value=None, visible=is_visible))
        return tuple(error_updates)


# --- Build Gradio App ---
# Define examples_display globally BEFORE it's used in functions
examples_display = None

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate", radius_size=gr.themes.sizes.radius_sm)) as demo:
    gr.Markdown("# Dynamic Hugging Face Model Runner")
    gr.Markdown("Enter a Hugging Face Model ID (e.g., `google/flan-t5-small`, `nlpconnect/vit-gpt2-image-captioning`), check details, load, and run inference.")

    # --- State ---
    temp_model_id_state = gr.State("")
    temp_task_state = gr.State("")

    # --- Row 1: Model Input ---
    with gr.Row():
        model_input = gr.Textbox(label="Hugging Face Model ID", placeholder="user/model-name", scale=4, container=False)
        check_button = gr.Button("Check Model", variant="secondary", scale=1)

    # --- Row 2: Confirmation Section ---
    with gr.Column(visible=False) as confirmation_section:
        confirmation_output = gr.Markdown()
        with gr.Row(): confirm_button = gr.Button("Confirm and Load Model", variant="primary")

    # --- Row 3: Inference Section ---
    input_placeholders = [gr.Textbox(visible=False, label=f"Input Placeholder {i+1}") for i in range(MAX_COMPONENTS)]
    output_placeholders = [gr.Textbox(visible=False, label=f"Output Placeholder {i+1}") for i in range(MAX_COMPONENTS)]

    with gr.Column(visible=False) as inference_section:
        inference_title = gr.Markdown("## Model Interface")
        with gr.Group():
            with gr.Row(equal_height=False):
                 with gr.Column(scale=1) as dynamic_inputs_column:
                     for comp in input_placeholders: comp
                 with gr.Column(scale=1) as dynamic_outputs_column:
                     for comp in output_placeholders: comp
        run_button = gr.Button("Run Inference", variant="primary")
        # --- Wrap Examples in a Group ---
        with gr.Group() as examples_group:
            # Assign to the global variable
            examples_display = gr.Examples(label="Examples", examples=[], inputs=[], examples_per_page=5)

    # --- Connect Actions ---
    check_button.click(
        fn=check_model, inputs=[model_input],
        # Outputs: confirm_section, confirm_output, state1, state2 (4 items)
        outputs=[confirmation_section, confirmation_output, temp_model_id_state, temp_task_state],
        show_progress="minimal"
    )

    confirm_button.click(
        fn=load_model_and_setup_ui, inputs=[temp_model_id_state, temp_task_state],
        # Outputs: confirm_section, infer_section, infer_title, 5*in, 5*out, examples_group (15 items)
        outputs=[
            confirmation_section, inference_section, inference_title,
            *input_placeholders, *output_placeholders,
            examples_group # Target the wrapper group
        ],
        show_progress="full" # Use progress indicator for loading
    )

    run_button.click(
        fn=run_inference, inputs=input_placeholders, outputs=output_placeholders,
        show_progress="full"
    )

    model_input.change(
         fn=handle_input_change, inputs=[],
         # Outputs: confirm_section, infer_section, confirm_output, infer_title, 5*in, 5*out, examples_group (16 items)
         outputs=[
            confirmation_section, inference_section, confirmation_output, inference_title,
            *input_placeholders, *output_placeholders,
            examples_group # Target the wrapper group
         ]
    )

# --- Launch the App ---
if __name__ == "__main__":
    # Note: Ensure 'accelerate' is installed if using GPU/MPS with device_map: pip install accelerate
    demo.queue().launch(debug=True)