from mlx_lm import generate, load

repo = "meta-llama/Llama-3.2-1B-Instruct"
model, tokenizer = load(repo, adapter_path="./adapters")  # pyright:ignore

prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGive feedback to the Helper's last response.\n\n### Input:\nHelper: Hi there, how are you doing today?\nSeeker: I am feeling really down.\nHelper: I'm sorry to hear you're feeling that way, what has you feeling this way?\nSeeker: My husband and I have 4 kids. One of those children is a child from his previous marriage. We constantly argue about how he favors that child over our other 3.\nHelper: It sounds like a hard situation to be in, and it must be tough for you. Can you describe how seeing him favor this child makes you feel?\n\n"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)  # pyright:ignore

text = generate(model, tokenizer, prompt=prompt, verbose=True)
print(text)
