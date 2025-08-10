from llama_cpp import Llama

# ⚠️ Make sure to replace this with the actual path to your downloaded GGUF file.
model_path = "./maral-model/Maral-7B-alpha-1-Q4_K_M.gguf" 

# Initialize the Llama model
# n_ctx is the context window size, which you can adjust.
# n_gpu_layers can be set to a number (e.g., -1 for all layers) to offload to the GPU.
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Adjust the context window as needed
        # For GPU acceleration, you can set the number of layers to offload.
        # e.g., n_gpu_layers=-1 will offload all layers to the GPU if available.
        # n_gpu_layers=0 will run on CPU only.
        n_gpu_layers=0,
        verbose=False # Set to True for more output from the Llama.cpp backend
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Define a prompt for the model.
# This model uses a specific chat template (`Guanaco format`).
# Using the correct format is crucial for getting good responses.
prompt = "### Human: cdn چیه?\n### Assistant:"

# Generate a response
print("\nGenerating response...")
output = llm(
    prompt,
    max_tokens=256,  # Maximum number of tokens to generate
    stop=["### Human:"],  # Stop generating when this token sequence is encountered
    echo=False,  # Don't echo the input prompt in the output
    temperature=0.7 # You can adjust this for more creative (higher) or more consistent (lower) answers.
)

# Extract and print the generated text
generated_text = output['choices'][0]['text'].strip()
print("Generated text:")
print(generated_text)