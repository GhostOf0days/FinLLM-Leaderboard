import re
import json
import threading
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from tqdm.auto import tqdm
from huggingface_hub import login

# Load config from config.json
with open("config.json", "r") as f:
    config = json.load(f)

# Model and dataset configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B"
DATASET_NAME = "ChanceFocus/flare-fiqasa"
ACCESS_TOKEN = config.get("huggingface_token", "")

def initialize_model():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        token=ACCESS_TOKEN,
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=ACCESS_TOKEN,
        trust_remote_code=True,
    )

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def zero_shot_prompt(example):
    """Construct standardized zero-shot prompt"""
    return f"""Analyze the sentiment of this financial text:
Text: {example['text']}
Options: {', '.join(example['choices'])}
Answer:"""

def generate_response(prompt, model, tokenizer, max_new_tokens=10):
    """Generate response with progress tracking"""
    # Tokenize with attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        input_ids=inputs.input_ids.to(model.device),
        attention_mask=inputs.attention_mask.to(model.device),
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    with tqdm(total=max_new_tokens, desc="Generating", unit="token") as pbar:
        for new_text in streamer:
            generated_text += new_text
            pbar.update(1)
    thread.join()
    return generated_text

def extract_answer(response):
    """Extract answer section from generated text"""
    lower = response.lower()
    idx = lower.find("answer:")
    if idx != -1:
        sec = response[idx + len("answer:"):].strip()
        expl = sec.lower().find("explanation:")
        return sec[:expl].strip() if expl != -1 else sec
    # fallback: no explicit "Answer:" so return full response for matching
    return response

def match_label(answer_section, choices):
    """Match extracted answer to valid choices"""
    if not answer_section:
        return None
    for choice in choices:
        if re.search(rf'\b{re.escape(choice)}\b', answer_section, re.IGNORECASE):
            return choice
    return None

def calculate_accuracy(predictions, references):
    """Calculate accuracy manually"""
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(references) if references else 0

def evaluate_model(model, tokenizer, dataset_split, num_samples=10):
    """Run evaluation with progress tracking"""
    predictions = []
    references = []
    results = []

    progress_bar = tqdm(total=num_samples, desc="Evaluating")

    for i in range(num_samples):
        ex = dataset_split[i]
        prompt = zero_shot_prompt(ex)
        response = generate_response(prompt, model, tokenizer)

        answer_section = extract_answer(response)
        pred_label = match_label(answer_section, ex['choices']) or "unknown"
        gold_label = ex['choices'][ex['gold']]

        # Convert to indices
        pred_index = ex['choices'].index(pred_label) if pred_label in ex['choices'] else -1
        predictions.append(pred_index)
        references.append(ex['gold'])

        # Store result
        results.append({
            'text': ex['text'],
            'response': response,
            'predicted': pred_label,
            'gold': gold_label,
            'correct': (pred_index == ex['gold'])
        })

        progress_bar.update(1)
        acc = calculate_accuracy(predictions, references)
        progress_bar.set_postfix({"accuracy": f"{acc:.2%}"})

    progress_bar.close()
    final_accuracy = calculate_accuracy(predictions, references)
    return {"accuracy": final_accuracy, "results": results}

if __name__ == "__main__":
    # Login to Hugging Face hub
    login(token=ACCESS_TOKEN)

    # Initialize model & tokenizer
    model, tokenizer = initialize_model()
    dataset = load_dataset(DATASET_NAME)

    # Run evaluation
    evaluation = evaluate_model(
        model,
        tokenizer,
        dataset["test"],
        num_samples=10
    )

    print(f"\nFinal Accuracy: {evaluation['accuracy']:.2%}")

    # Print a few example results
    print("\nResults:")
    for i, result in enumerate(evaluation['results']):
        print(f"Example {i+1}: {result['text'][:50]}...")
        print(f"Predicted: {result['predicted']}, Gold: {result['gold']}")
        print(f"Correct: {'✓' if result['correct'] else '✗'}")
        print()