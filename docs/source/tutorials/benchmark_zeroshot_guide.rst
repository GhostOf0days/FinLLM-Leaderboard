.. _zero_shot_llama:

Benchmark Llama-3.2 on Financial Sentiment Analysis (zeroshot)
=====================================================================

.. contents:: Table of Contents
   :local:

Overview
--------
This guide shows how to benchmark Llama-3.2-3B model in a **Zero-Shot** setting:

1. Configure API access using config.json
2. Load model from Hugging Face
3. Prompt the model for financial sentiment analysis
4. Evaluate accuracy on the FLARE-FIQASA dataset

Prerequisites
-------------

1. config.json containing your Hugging Face Access Token:

   .. code-block:: json

       {
         "huggingface_token": "your_huggingface_token_here"
       }

   .. note::
       Create this file and add your real Hugging Face token before starting.

2. Request model access at Hugging Face:
   - Create a token at `huggingface.co/settings/tokens <https://huggingface.co/settings/tokens>`_
   - Request access for `Llama-3.2-3B <https://huggingface.co/meta-llama/Llama-3.2-3B>`_

3. Dataset structure ``flare-fiqasa`` `<https://huggingface.co/datasets/ChanceFocus/flare-fiqasa>`_:

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

4. Install dependencies:

   .. code-block:: bash

      pip install accelerate transformers datasets scikit-learn tqdm torch huggingface_hub

Tutorial
--------

1. Import Libraries

   .. code-block:: python

      import re
      import json
      import threading
      from datasets import load_dataset
      from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
      from tqdm.auto import tqdm
      from huggingface_hub import login

   .. note:: 
       Import libraries for loading models, processing datasets, and tracking progress.

2. Configuration Setup

   .. code-block:: python

      # Load config from config.json
      with open("config.json", "r") as f:
          config = json.load(f)

      # Model and dataset configuration
      MODEL_NAME = "meta-llama/Llama-3.2-3B"
      DATASET_NAME = "ChanceFocus/flare-fiqasa"
      ACCESS_TOKEN = config.get("huggingface_token", "")

   .. note::
       Load your Hugging Face token from config.json and set up the model and dataset paths.

3. Hugging Face Authentication

   .. code-block:: python
   
      # Login to Hugging Face hub
      login(token=ACCESS_TOKEN)

4. Model Initialization

   .. code-block:: python

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

   .. note::
       Load the Llama-3.2 model and tokenizer from Hugging Face using your access token.

5. Zero-Shot Prompt Template

   .. code-block:: python

      def zero_shot_prompt(example):
          """Construct standardized zero-shot prompt"""
          return f"""Analyze the sentiment of this financial text:
      Text: {example['text']}
      Options: {', '.join(example['choices'])}
      Answer:"""

   .. note::
       Create a simple prompt that asks the model to analyze sentiment without any examples.

6. Generation Function

   .. code-block:: python

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

   .. note::
       Run the model in a separate thread to generate a response while showing a progress bar.

7. Answer Extraction

   .. code-block:: python

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

   .. note::
       Extract and parse the model's answer, handling different response formats.

8. Accuracy Calculation

   .. code-block:: python

      def calculate_accuracy(predictions, references):
          """Calculate accuracy manually"""
          correct = sum(1 for p, r in zip(predictions, references) if p == r)
          return correct / len(references) if references else 0

   .. note::
       A simple function to calculate the percentage of correct predictions.

9. Evaluation Function

   .. code-block:: python

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
          
          # Calculate final accuracy
          final_accuracy = calculate_accuracy(predictions, references)
          
          return {"accuracy": final_accuracy, "results": results}

   .. note::
       Test the model on examples from the dataset and calculate the accuracy.

10. Main Execution

    .. code-block:: python

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

   .. note::
       Run the full evaluation pipeline and report the accuracy with example results.

Running the Tutorial
--------------------

1. Create a config.json file with your Hugging Face token
2. Ensure GPU availability for model inference
3. Save code as ``benchmark_llama_zeroshot.py``
4. Run with ``python benchmark_llama_zeroshot.py``

Example Output
----------------

.. code-block:: text

    Loading model...
    Loading tokenizer...
    Evaluating: 100%|████████████████| 10/10 [00:12<00:00,  1.22s/it, accuracy=70.00%]

    Final Accuracy: 70.00%
    
    Results:
    Example 1: Whats up with $LULU? Numbers looked good...
    Predicted: neutral, Gold: neutral
    Correct: ✓
    
    Example 2: $COST thesis: the most bullet-proof retailer...
    Predicted: positive, Gold: positive
    Correct: ✓

Notes
-----
- **Zero-shot learning**: Testing models without any examples or fine-tuning
- **Threading**: Running generation in a separate thread for better UX
- **Answer extraction**: Parsing model outputs to find the sentiment label
- **Temperature**: Controls randomness in model outputs
- **Max Tokens**: Limits the length of generated responses

Additional Resources
----------------------
- `Llama 3.2 Model Card <https://huggingface.co/meta-llama/Llama-3.2-3B>`_
- `FLARE-FIQASA Dataset <https://huggingface.co/datasets/ChanceFocus/flare-fiqasa>`_
- `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`_