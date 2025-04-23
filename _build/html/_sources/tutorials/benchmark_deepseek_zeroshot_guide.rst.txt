.. _deepseek_zero_shot:

Benchmark DeepSeek on Financial Tasks
================================================

This guide demonstrates how to benchmark DeepSeek models in a **Zero-Shot** setting using their API:

1. **Configure** API access
2. **Prompt** the model with financial sentiment analysis questions
3. **Evaluate** accuracy against the FLARE-FIQASA dataset

Prerequisites
-------------

1. **DeepSeek API Key**:
   - Follow :ref:`deepseek_api_setup` to obtain credentials

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

      pip install openai datasets evaluate tqdm

Tutorial
--------

1. Import Libraries

   .. code-block:: python

      from openai import OpenAI
      from datasets import load_dataset
      import evaluate
      from tqdm.auto import tqdm

2. Configuration Setup

   .. code-block:: python

      # API configuration
      API_KEY = "your_deepseek_api_key"  # Replace with actual key
      MODEL_NAME = "deepseek-chat"  # Alternatives: deepseek-reasoner
      DATASET_NAME = "ChanceFocus/flare-fiqasa"

3. Initialize Client

   .. code-block:: python

      def initialize_client():
          return OpenAI(
              api_key=API_KEY,
              base_url="https://api.deepseek.com"
          )

4. Zero-Shot Prompt Template

   .. code-block:: python

      def zero_shot_prompt(example):
          """Construct standardized zero-shot prompt"""
          return f"""Analyze the sentiment of this financial text. Respond ONLY with the label:
      Text: {example['text']}
      Options: {', '.join(example['choices'])}
      Label:"""

5. API Call Function

   .. code-block:: python

      def get_api_response(client, prompt):
          """Get response from DeepSeek API"""
          try:
              response = client.chat.completions.create(
                  model=MODEL_NAME,
                  messages=[
                      {"role": "system", "content": "You are a financial analyst."},
                      {"role": "user", "content": prompt}
                  ],
                  temperature=0,
                  max_tokens=10
              )
              return response.choices[0].message.content.strip()
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

7. Evaluation Function

   .. code-block:: python

      def evaluate_model(client, dataset_split, num_samples=10):
          """Run evaluation with progress tracking"""
          accuracy = evaluate.load("accuracy")
          predictions = []
          references = []

          progress_bar = tqdm(total=num_samples, desc="Evaluating")

          for i in range(num_samples):
              ex = dataset_split[i]
              prompt = zero_shot_prompt(ex)
              response = get_api_response(client, prompt)
              pred_label = validate_response(response, ex['choices'])
              gold_label = ex['choices'][ex['gold']]

              predictions.append(ex['choices'].index(pred_label) if pred_label in ex['choices'] else -1
              references.append(ex['gold'])

              progress_bar.update(1)
              current_acc = accuracy.compute(predictions=predictions, references=references)['accuracy']
              progress_bar.set_postfix({"accuracy": f"{current_acc:.2%}"})

          progress_bar.close()
          return accuracy.compute(predictions=predictions, references=references)

8. Main Execution

   .. code-block:: python

      if __name__ == "__main__":
          client = initialize_client()
          dataset = load_dataset(DATASET_NAME)

          results = evaluate_model(
              client,
              dataset["test"],
              num_samples=10
          )

          print(f"\nFinal Accuracy: {results['accuracy']:.2%}")

Running the Tutorial
--------------------

1. Replace ``your_deepseek_api_key`` with your actual API key
2. Save code as ``deepseek_zero_shot.py``
3. Run with ``python deepseek_zero_shot.py``

Notes
-----
- **Rate Limits**: DeepSeek API has default 50 RPM limit
- **Model Selection**:
   - ``deepseek-chat`` for general financial analysis
   - ``deepseek-reasoner`` for complex reasoning tasks
- **Prompt Design**: Explicit instruction ("Respond ONLY with the label") improves consistency
- **Error Handling**: API calls include basic exception handling

Additional Resources
--------------------
- `DeepSeek API Documentation <https://api.deepseek.com/docs>`_
- `FLARE-FIQASA Dataset Paper <https://arxiv.org/abs/2308.00075>`_