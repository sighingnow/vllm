import os

from vllm import LLM, SamplingParams

os.environ["VLLM_ATTENTION_BACKEND"] = "DUAL_CHUNK_FLASH_ATTN"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

with open(os.path.expanduser("~/vllm/64k.txt")) as f:
    prompt = f.read()

# Sample prompts.
prompts = [
    prompt,
]
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.7,
    top_k=20,
    detokenize=True,
)

# Create an LLM.
llm = LLM(model=os.path.expanduser("~/models/qwen2.5-14b-1m-1231/"),
          max_model_len=1048576,
          tensor_parallel_size=4,
          enforce_eager=True)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
