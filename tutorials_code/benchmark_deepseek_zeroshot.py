import json
import requests
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

# Load API keys from config.json
with open('config.json') as f:
    config = json.load(f)

# API configuration
MODEL_NAME = "deepseek-chat"  # Alternatives: deepseek-reasoner
DATASET_NAME = "ChanceFocus/flare-fiqasa"

# Set up DeepSeek headers
deepseek_headers = {
    "Authorization": f"Bearer {config['deepseek_api_key']}",
    "Content-Type": "application/json"
}

def zero_shot_prompt(example):
    """Construct standardized zero-shot prompt"""
    return f"""Analyze the sentiment of this financial text. Respond ONLY with the label:
Text: {example['text']}
Options: {', '.join(example['choices'])}
Label:"""

def get_api_response(prompt, model=MODEL_NAME):
    """Get response from DeepSeek API using requests"""
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 10
    }
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions", 
            json=payload, 
            headers=deepseek_headers
        )
        result = response.json()
        if 'choices' not in result:
            raise ValueError(f"DeepSeek API error: {result.get('error', result)}")
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"API Error: {e}")
        return ""

def validate_response(response, choices):
    """Match API response to valid choices"""
    response = response.lower()
    for choice in choices:
        if choice in response:
            return choice
    return None

def calculate_accuracy(predictions, references):
    """Calculate accuracy manually"""
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(references) if len(references) > 0 else 0

def evaluate_model(dataset_split, num_samples=10):
    """Run evaluation with progress tracking"""
    predictions = []
    references = []
    results = []

    progress_bar = tqdm(total=num_samples, desc="Evaluating")

    for i in range(num_samples):
        ex = dataset_split[i]
        prompt = zero_shot_prompt(ex)
        response = get_api_response(prompt)
        pred_label = validate_response(response, ex['choices'])
        gold_label = ex['choices'][ex['gold']]
        
        # For display purposes
        pred_index = ex['choices'].index(pred_label) if pred_label in ex['choices'] else -1
        
        # Store results
        predictions.append(pred_index)
        references.append(ex['gold'])
        
        # Record individual result
        results.append({
            'text': ex['text'],
            'model_response': response,
            'predicted': pred_label,
            'gold': gold_label,
            'correct': pred_index == ex['gold']
        })

        progress_bar.update(1)
        current_acc = calculate_accuracy(predictions, references)
        progress_bar.set_postfix({"accuracy": f"{current_acc:.2%}"})

    progress_bar.close()
    
    # Calculate final accuracy
    final_accuracy = calculate_accuracy(predictions, references)
    
    return {"accuracy": final_accuracy, "results": results}

if __name__ == "__main__":
    dataset = load_dataset(DATASET_NAME)

    evaluation_results = evaluate_model(
        dataset["test"],
        num_samples=10
    )

    print(f"\nFinal Accuracy: {evaluation_results['accuracy']:.2%}")
    
    # Print individual results
    print("\nIndividual Results:")
    for i, result in enumerate(evaluation_results['results']):
        print(f"{i+1}. Text: {result['text'][:50]}...")
        print(f"   Model response: {result['model_response']}")
        print(f"   Predicted: {result['predicted']}, Gold: {result['gold']}")
        print(f"   Correct: {'✓' if result['correct'] else '✗'}")
        print()