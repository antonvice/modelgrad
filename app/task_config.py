# app/task_config.py
import gradio as gr

# --- Configuration ---
# Map pipeline tasks to Gradio input/output components and preprocessing/postprocessing logic
TASK_IO_MAP = {
    "text-generation": {
        "inputs": lambda: [gr.Textbox(label="Input Text", lines=5, interactive=True)],
        "outputs": lambda: [gr.Textbox(label="Generated Text", lines=5)],
        "preprocess": lambda *args: args[0], # Input is text
        "postprocess": lambda result: result[0]['generated_text'] if result and isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0] else str(result),
        "example_inputs": ["Once upon a time"],
        "pipeline_args": {"max_new_tokens": 100} # Example default args
    },
    "text-classification": {
        "inputs": lambda: [gr.Textbox(label="Text to Classify", lines=3, interactive=True)],
        "outputs": lambda: [gr.Label(label="Classification Result")], # Label is good for dict output
        "preprocess": lambda *args: args[0],
        "postprocess": lambda result: {item['label']: round(item['score'], 3) for item in result} if isinstance(result, list) else str(result), # Format for Label
        "example_inputs": ["This movie was great!"],
         "pipeline_args": {}
    },
    "summarization": {
        "inputs": lambda: [gr.Textbox(label="Text to Summarize", lines=10, interactive=True)],
        "outputs": lambda: [gr.Textbox(label="Summary", lines=5)],
        "preprocess": lambda *args: args[0],
        "postprocess": lambda result: result[0]['summary_text'] if result and isinstance(result, list) and len(result) > 0 and 'summary_text' in result[0] else str(result),
        "example_inputs": ["Paris is the capital and most populous city of France..."],
         "pipeline_args": {"min_length": 10, "max_length": 50}
    },
    "translation_xx_to_yy": { # Placeholder, specific language pairs needed usually
        "inputs": lambda: [gr.Textbox(label="Text to Translate", lines=5, interactive=True)],
        "outputs": lambda: [gr.Textbox(label="Translated Text", lines=5)],
        "preprocess": lambda *args: args[0],
        "postprocess": lambda result: result[0]['translation_text'] if result and isinstance(result, list) and len(result) > 0 and 'translation_text' in result[0] else str(result),
        "example_inputs": ["Hello, world!"],
         "pipeline_args": {}
    },
     "fill-mask": {
        "inputs": lambda: [gr.Textbox(label="Text with <mask> token", lines=2, interactive=True)],
        "outputs": lambda: [gr.Label(label="Top Fill Options")], # Using Label to show top k nicely
        "preprocess": lambda *args: args[0],
        "postprocess": lambda result: {item['token_str']: round(item['score'], 3) for item in result} if isinstance(result, list) else str(result), # Format for Label
        "example_inputs": ["The capital of France is <mask>."],
        "pipeline_args": {"top_k": 5}
    },
    "question-answering": {
        "inputs": lambda: [
            gr.Textbox(label="Context", lines=7, interactive=True),
            gr.Textbox(label="Question", lines=2, interactive=True)
        ],
        "outputs": lambda: [
            gr.Textbox(label="Answer", lines=2),
            gr.Label(label="Score") # Show confidence score
        ],
        "preprocess": lambda context, question: {"context": context, "question": question},
        "postprocess": lambda result: (result.get('answer', 'N/A'), {"score": round(result.get('score', 0.0), 4)}) if isinstance(result, dict) else ("Error processing result", {"score": 0.0}),
        "example_inputs": ["The Apollo program was the third United States human spaceflight program.", "What program sent humans to the moon?"],
         "pipeline_args": {}
    },
    "image-classification": {
        "inputs": lambda: [gr.Image(type="pil", label="Image to Classify", interactive=True)],
        "outputs": lambda: [gr.Label(label="Classification Result")],
        "preprocess": lambda *args: args[0], # Input is PIL image
        "postprocess": lambda result: {item['label']: round(item['score'], 3) for item in result} if isinstance(result, list) else str(result), # Format for Label
        "example_inputs": ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"], # URL or local path works
         "pipeline_args": {}
    },
     "image-to-text": {
        "inputs": lambda: [gr.Image(type="pil", label="Image to Caption", interactive=True)],
        "outputs": lambda: [gr.Textbox(label="Generated Caption", lines=3)],
        "preprocess": lambda *args: args[0],
        "postprocess": lambda result: result[0]['generated_text'] if result and isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0] else str(result),
        "example_inputs": ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"],
        "pipeline_args": {}
    },
     "text-to-image": {
        "inputs": lambda: [gr.Textbox(label="Prompt for Image Generation", lines=3, interactive=True)],
        "outputs": lambda: [gr.Image(type="pil", label="Generated Image")],
        "preprocess": lambda *args: args[0],
        "postprocess": lambda result: result['images'][0] if result and 'images' in result and result['images'] else None, # Assumes diffusers format output
         "example_inputs": ["A photo of an astronaut riding a horse on the moon"],
        "pipeline_args": {} # May need specific diffusers args
    },
    # Add more tasks here following the pattern
    # e.g., "zero-shot-classification", "object-detection", "automatic-speech-recognition" etc.
}