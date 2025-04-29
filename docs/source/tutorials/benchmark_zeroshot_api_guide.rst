.. _zero_shot_openai:

Benchmark GPT-4o on Financial Sentiment Analysis (zeroshot)
=============================================================

.. contents:: Table of Contents
   :local:

This guide shows how to benchmark an OpenAI model in a **Zero-Shot** setting:

1. Configure API access using config.json
2. Prompt the model with financial multiple-choice questions
3. Evaluate accuracy and display results

Prerequisites
-------------
1. config.json containing your OpenAI API key:

   .. code-block:: json

       {
         "openai_api_key": "your-openai-key-here"
       }

   .. note::
      Create this file and add your real API key before starting.

2. Sample financial questions in a list (or alternatively a CSV file):

   .. list-table:: Example Questions
      :header-rows: 1
      :widths: 40 20 20 20 10

      * - Question
        - Choice A
        - Choice B
        - Choice C
        - Answer
      * - "Sammy Sneadle, CFA, is the founder of the Everglades Fund. How did he violate the standard by not disclosing back-tested data?"
        - "He did not disclose the use of back-tested data."
        - "He failed to deduct fees before returns."
        - "He did not show a weighted composite of similar portfolios."
        - "A"

   .. note::
      You can use these examples or add your own financial questions.

3. Install the necessary libraries:

   .. code-block:: bash

      pip install openai pandas tqdm

   .. note::
      OpenAI provides access to the API, pandas helps with data handling, and tqdm creates progress bars.

Tutorial
--------
1. Import Libraries

   .. code-block:: python

      import json
      import pandas as pd
      from tqdm import tqdm
      from openai import OpenAI

   .. note::
      Standard imports for working with JSON files, data processing, and the OpenAI API.

2. Load API Configuration

   .. code-block:: python
      
      # Load API keys from config.json
      with open("config.json", "r") as f:
          config = json.load(f)
      
      # Initialize OpenAI client
      client = OpenAI(api_key=config["openai_api_key"])

   .. note::
      This reads your API key and sets up the OpenAI client.

3. Define Questions

   .. code-block:: python

      # Sample financial questions
      questions = [
          {
              "question": "Sammy Sneadle, CFA, is the founder of the Everglades Fund. How did he violate the standard by not disclosing back-tested data?",
              "choiceA": "He did not disclose the use of back-tested data.",
              "choiceB": "He failed to deduct fees before returns.",
              "choiceC": "He did not show a weighted composite of similar portfolios.",
              "answer": "A"
          },
          {
              "question": "What is the primary goal of corporate finance?",
              "choiceA": "Maximizing shareholder value",
              "choiceB": "Minimizing operational costs",
              "choiceC": "Increasing market share",
              "answer": "A"
          }
          # Add more questions as needed
      ]

   .. note::
      Create a list of question dictionaries with our test cases.

4. Zero-Shot Inference Function

   .. code-block:: python

      def ask_openai(question, choiceA, choiceB, choiceC, model="gpt-4o"):
          """Generate a zero-shot response to a financial question"""
          system_prompt = (
              "You are a CFA (chartered financial analyst) taking a test. "
              "You will be given a question with three possible answers (A, B, and C). "
              "Provide only the letter for the correct choice (A, B, or C)."
          )
          
          user_prompt = (
              f"Question:\n{question}\n\n"
              f"A. {choiceA}\n"
              f"B. {choiceB}\n"
              f"C. {choiceC}\n\n"
              "Which choice is correct? Answer with just the letter A, B, or C."
          )
          
          try:
              response = client.chat.completions.create(
                  model=model,
                  messages=[
                      {"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}
                  ],
                  temperature=0,
                  max_tokens=10
              )
              return response.choices[0].message.content.strip()
          except Exception as e:
              print(f"Error: {e}")
              return f"Error: {e}"

   .. note::
      This function asks GPT-4o to answer a financial question and return only a letter choice.

5. Evaluation Function

   .. code-block:: python

      def evaluate_questions(questions_list, model="gpt-4o"):
          """Evaluate model on a list of questions"""
          results = []
          correct_count = 0
          
          for q in tqdm(questions_list, desc=f"Testing {model}"):
              question = q["question"]
              choiceA = q["choiceA"]
              choiceB = q["choiceB"] 
              choiceC = q["choiceC"]
              expected = q["answer"].upper().strip()
              
              # Get model response
              response = ask_openai(
                  question=question,
                  choiceA=choiceA,
                  choiceB=choiceB,
                  choiceC=choiceC,
                  model=model
              )
              
              # Extract just the letter (A, B, or C)
              generated = response.upper().strip()
              if len(generated) > 1:
                  # If response contains more than just the letter, try to extract the letter
                  if "A" in generated:
                      generated = "A"
                  elif "B" in generated:
                      generated = "B" 
                  elif "C" in generated:
                      generated = "C"
                  else:
                      generated = "INVALID"
              
              # Check if correct
              is_correct = generated == expected
              if is_correct:
                  correct_count += 1
              
              # Store result
              results.append({
                  "question": question[:50] + "..." if len(question) > 50 else question,
                  "model_response": response,
                  "generated": generated,
                  "expected": expected,
                  "correct": is_correct
              })
          
          # Calculate accuracy
          accuracy = correct_count / len(questions_list) if questions_list else 0
          
          return {
              "accuracy": accuracy,
              "results": results
          }

   .. note::
      Loops through each question, gets the model's answer, and tracks if it's right or wrong.

6. Main Execution

   .. code-block:: python

      if __name__ == "__main__":
          # Run the evaluation
          evaluation = evaluate_questions(questions, model="gpt-4o")
          
          # Print results
          print(f"\nAccuracy: {evaluation['accuracy']:.2%}")
          
          print("\nResults:")
          for i, r in enumerate(evaluation['results']):
              print(f"Q{i+1}: {r['question']}")
              print(f"Response: {r['model_response']}")
              print(f"Expected: {r['expected']}, Got: {r['generated']}")
              print("✓" if r['correct'] else "✗")
              print()

   .. note::
      Runs the evaluation and shows the accuracy along with each question's result.

Running the Tutorial
--------------------
1. Create a config.json file with your OpenAI API key
2. Save the code as ``benchmark_gpt4o_zeroshot.py``
3. Run with ``python benchmark_gpt4o_zeroshot.py``

Key Takeaways
-------------
This tutorial demonstrates how to benchmark an OpenAI model in a zero-shot setting for financial tasks. It shows how to load API keys, define questions, generate responses, and evaluate performance.

Notes that you can refer back to later
--------------------------------------

- **Zero-shot** means that you prompt a model with just the question and no examples.
- **temperature** determines the randomness of the model response. The closer to 0, the more deterministic and consistent the model response is.
- **max_tokens** determines the maximum length of the output of the model.
- **system_prompt** determines the behavior/domain context the model should follow.
- **prompt** is the actual question the user gives to the model. It is also called a user prompt.

Additional Resources
----------------------
- `OpenAI API Documentation <https://platform.openai.com/docs/api-reference>`_
- `GPT-4o Model Information <https://platform.openai.com/docs/models/gpt-4o>`_
