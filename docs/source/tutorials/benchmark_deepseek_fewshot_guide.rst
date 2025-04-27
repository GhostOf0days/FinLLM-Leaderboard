.. _deepseek_few_shot:

Benchmark DeepSeek on Financial Sentiment Analysis (few-shot)
========================================================

This guide demonstrates how to benchmark DeepSeek models in a **Few-Shot** setting using their API:

1. Configure API access using config.json
2. Provide examples to the model
3. Prompt the model with financial sentiment analysis questions
4. Evaluate accuracy against the FLARE-FIQASA dataset

Prerequisites
-------------

1. config.json containing your DeepSeek API key:

   .. code-block:: json

       {
         "deepseek_api_key": "your_deepseek_api_key"
       }

2. Dataset structure ``flare-fiqasa`` `<https://huggingface.co/datasets/ChanceFocus/flare-fiqasa>`_:

   .. list-table:: Example Dataset Entry
      :header-rows: 1
      :widths: 20 20 20 20 20

      * - text
        - choices
        - gold
        - answer
        - id
      * - "Whats up with $LULU? Numbers looked good..."
        - ["negative", "positive", "neutral"]
        - 2
        - "neutral"
        - "fiqasa0"

3. Install dependencies:

   .. code-block:: bash

      pip install requests datasets tqdm

Tutorial
--------

1. Import Libraries

   .. code-block:: python

      import json
      import requests
      from datasets import load_dataset
      from tqdm.auto import tqdm

2. Configuration Setup

   .. code-block:: python

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

3. Define Few-Shot Examples

   .. code-block:: python

      # Define examples for few-shot learning
      few_shot_examples = [
          {
              "text": "Apple is doing great with the new iPhone release, stock is up 5%.",
              "choices": ["negative", "positive", "neutral"],
              "answer": "positive"
          },
          {
              "text": "Tesla missed earnings expectations and reported a decline in margins.",
              "choices": ["negative", "positive", "neutral"],
              "answer": "negative"
          },
          {
              "text": "IBM reported Q3 results in line with analyst expectations.",
              "choices": ["negative", "positive", "neutral"],
              "answer": "neutral"
          },
          {
              "text": "The market traded sideways today with no major movements.",
              "choices": ["negative", "positive", "neutral"],
              "answer": "neutral"
          }
      ]

4. Few-Shot Prompt Template

   .. code-block:: python

      def few_shot_prompt(example, examples):
          """Construct few-shot prompt with examples"""
          prompt = "Analyze the sentiment of financial texts. Choose negative, positive, or neutral.\n\n"
          
          # Add examples
          for i, ex in enumerate(examples):
              prompt += f"Example {i+1}:\n"
              prompt += f"Text: {ex['text']}\n"
              prompt += f"Options: {', '.join(ex['choices'])}\n"
              prompt += f"Label: {ex['answer']}\n\n"
          
          # Add the test example
          prompt += f"Now analyze this new text:\n"
          prompt += f"Text: {example['text']}\n"
          prompt += f"Options: {', '.join(example['choices'])}\n"
          prompt += f"Label:"
          
          return prompt

5. API Call Function

   .. code-block:: python

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

6. Answer Validation

   .. code-block:: python

      def validate_response(response, choices):
          """Match API response to valid choices"""
          response = response.lower()
          for choice in choices:
              if choice in response:
                  return choice
          return None

7. Accuracy Calculation

   .. code-block:: python

      def calculate_accuracy(predictions, references):
          """Calculate accuracy manually"""
          correct = sum(1 for p, r in zip(predictions, references) if p == r)
          return correct / len(references) if len(references) > 0 else 0

8. Evaluation Function

   .. code-block:: python

      def evaluate_model(dataset_split, examples, num_samples=10):
          """Run evaluation with few-shot examples"""
          predictions = []
          references = []
          results = []

          progress_bar = tqdm(total=num_samples, desc="Evaluating")

          for i in range(num_samples):
              ex = dataset_split[i]
              prompt = few_shot_prompt(ex, examples)
              response = get_api_response(prompt)
              pred_label = validate_response(response, ex['choices'])
              gold_label = ex['choices'][ex['gold']]
              
              # Store predictions
              predictions.append(ex['choices'].index(pred_label) if pred_label in ex['choices'] else -1)
              references.append(ex['gold'])
              
              # Store detailed results
              results.append({
                  'text': ex['text'],
                  'model_response': response,
                  'predicted': pred_label,
                  'gold': gold_label,
                  'correct': (pred_label == gold_label)
              })

              progress_bar.update(1)
              current_acc = calculate_accuracy(predictions, references)
              progress_bar.set_postfix({"accuracy": f"{current_acc:.2%}"})

          progress_bar.close()
          
          # Calculate final accuracy
          final_accuracy = calculate_accuracy(predictions, references)
          
          return {"accuracy": final_accuracy, "results": results}

9. Main Execution

   .. code-block:: python

      if __name__ == "__main__":
          # Load dataset
          dataset = load_dataset(DATASET_NAME)
          
          # Run evaluation using few-shot learning
          evaluation = evaluate_model(
              dataset["test"],
              few_shot_examples,
              num_samples=10
          )
          
          # Print results
          print(f"\nAccuracy: {evaluation['accuracy']:.2%}")
          
          print("\nResults:")
          for i, result in enumerate(evaluation['results']):
              print(f"Example {i+1}: {result['text'][:50]}...")
              print(f"Predicted: {result['predicted']}, Gold: {result['gold']}")
              print(f"Correct: {'✓' if result['correct'] else '✗'}")
              print()

Running the Tutorial
--------------------

1. Create a config.json file with your DeepSeek API key
2. Save code as ``benchmark_deepseek_fewshot.py``
3. Run with ``python benchmark_deepseek_fewshot.py``

Notes
-----

- **Few-shot learning** means providing examples to the model before asking it to solve a new problem
- **Temperature** determines the randomness of the model response, with 0 producing deterministic outputs
- **Max Tokens** limits the length of the model's response
- **System Prompt** guides the model's behavior and provides context

Additional Resources
-------------------
- `DeepSeek API Documentation <https://api.deepseek.com/docs>`_
- `FLARE-FIQASA Dataset <https://huggingface.co/datasets/ChanceFocus/flare-fiqasa>`_ 