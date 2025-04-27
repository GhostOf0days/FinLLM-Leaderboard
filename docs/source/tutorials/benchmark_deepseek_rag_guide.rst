.. _deepseek_rag:

Benchmark DeepSeek on Financial Questions with RAG
==================================================

This guide demonstrates how to benchmark DeepSeek models with **Retrieval-Augmented Generation (RAG)** using their API:

1. Configure API access using config.json
2. Set up a FAISS index with financial documents
3. Retrieve relevant context for each question
4. Prompt the model with financial questions and RAG context
5. Evaluate performance on CFA exam questions from Kaplan Schweser

Prerequisites
-------------

1. config.json containing your API keys:

   .. code-block:: json

       {
         "deepseek_api_key": "your_deepseek_api_key",
         "openai_api_key": "your_openai_api_key_for_embeddings"
       }

2. RAG resources:
   - A FAISS index of CFA materials
   - Corresponding text chunks
   - Sample questions from Kaplan Schweser

3. Install dependencies:

   .. code-block:: bash

      pip install requests openai faiss-cpu numpy tqdm pickle5

Tutorial
--------

1. Import Libraries

   .. code-block:: python

      import json
      import requests
      import numpy as np
      import os
      import faiss
      import pickle
      from tqdm import tqdm
      from openai import OpenAI

2. Configuration Setup

   .. code-block:: python

      # Load API keys from config.json
      with open('config.json') as f:
          config = json.load(f)
      
      # API configuration
      DEEPSEEK_MODEL = "deepseek-chat"  # Alternatives: deepseek-reasoner
      
      # Set up DeepSeek headers
      deepseek_headers = {
          "Authorization": f"Bearer {config['deepseek_api_key']}",
          "Content-Type": "application/json"
      }
      
      # Initialize OpenAI client for embeddings
      openai_client = OpenAI(api_key=config["openai_api_key"])
      
      # RAG resources paths
      CFA_RAG_DIR = './RAG/CFA_RAG'
      CFA_INDEX_PATH = os.path.join(CFA_RAG_DIR, 'faiss_index.idx')
      CFA_CHUNKS_PATH = os.path.join(CFA_RAG_DIR, 'chunks.pkl')

3. RAG Components

   .. code-block:: python

      def load_index_and_chunks(index_file=CFA_INDEX_PATH, chunks_file=CFA_CHUNKS_PATH):
          """Loads the FAISS index and text chunks from disk."""
          print(f"Loading index from {index_file}")
          index = faiss.read_index(index_file)

          print(f"Loading chunks from {chunks_file}")
          with open(chunks_file, 'rb') as f:
              chunks = pickle.load(f)

          print(f"Loaded index with {index.ntotal} vectors and {len(chunks)} chunks")
          return index, chunks

      def embed_query(query, client=openai_client, model="text-embedding-ada-002"):
          """Generates an embedding for the query text."""
          response = client.embeddings.create(
              input=[query],
              model=model
          )
          return response.data[0].embedding

      def retrieve_chunks(query, index, chunks, k=5):
          """Retrieves the most relevant text chunks for a given query."""
          query_embedding = embed_query(query)
          query_vector = np.array([query_embedding]).astype('float32')

          # Search the index
          _, indices = index.search(query_vector, k)
          results = [chunks[i] for i in indices[0]]

          return results

4. Sample CFA Questions from Kaplan Schweser

   .. code-block:: python
   
      # Sample questions from Kaplan Schweser CFA materials
      kaplan_schweser_questions = [
          {
              "question": "If the market yield does not change, the price of a Treasury bill:",
              "choices": {
                  "A": "Will increase as the bill approaches maturity.", 
                  "B": "Will decrease as the bill approaches maturity.", 
                  "C": "Stay the same as the bill approaches maturity."
              },
              "answer": "A",
              "explanation": "Treasury bills are discount instruments. As the bond approaches maturity, the price would increase."
          },
          {
              "question": "Which of the following is closest to the percentage price change of a bond for a 20 basis point increase in the yield if the bond's duration is 8.54 and the convexity is 58.66?",
              "choices": {
                  "A": "–1.696%.", 
                  "B": "–1.708%.", 
                  "C": "–1.720%."
              },
              "answer": "A",
              "explanation": "Percentage price change = (- Duration x change in yield) + (½ x Convexity x change in yield²) = – 8.54 x 0.002 + ½ x 58.66 x 0.002² = –0.01696 x 100 = –1.696%"
          },
          {
              "question": "Evaluate the following statements. Statement 1: \"A putable bond exhibits negative convexity at low yields and positive convexity at high yields.\" Statement 2: \"Effective duration measures the sensitivity of a bond's price to changes in its yield to maturity.\"",
              "choices": {
                  "A": "Both statements are correct.", 
                  "B": "Exactly one statement is correct.", 
                  "C": "None of the statements are correct."
              },
              "answer": "C",
              "explanation": "Statement 1 is incorrect. It describes a callable bond, not a putable bond. Statement 2 is incorrect. Effective duration measures sensitivity to changes in the benchmark yield curve, not yield to maturity."
          },
          {
              "question": "The nominal GDP is equal to $55,240,000 and the real GDP is equal to $52,040,000. The GDP deflator is closest to:",
              "choices": {
                  "A": "94", 
                  "B": "100", 
                  "C": "106"
              },
              "answer": "C",
              "explanation": "GDP deflator = (Nominal GDP / Real GDP) × 100 = ($55,240,000 / $52,040,000) × 100 = 106.1"
          }
      ]

5. Prompt Construction

   .. code-block:: python

      def make_prompt(question, choices, context=""):
          """Creates a prompt for answering multiple choice questions with RAG."""
          formatted_choices = ""
          for letter, choice_text in choices.items():
              formatted_choices += f"{letter}. {choice_text}\n"

          return f"""You are a financial analyst answering CFA exam questions. Your response will be used to benchmark your ability in financial analysis. It is crucial that your answer adheres strictly to the format of the question provided.

      Instructions for Answering Multiple-Choice Question:
      - Format: Provide your answer as just the letter (A, B, or C).
      - No Explanations: Do not include any explanations, reasoning, or additional text.

      Context from financial materials:
      {context}

      Question: {question}
      {formatted_choices}

      Which of the choices is most likely correct? Answer with only the letter."""

6. DeepSeek API Call

   .. code-block:: python

      def ask_deepseek(prompt, model=DEEPSEEK_MODEL):
          """Query DeepSeek API with prompt"""
          payload = {
              "model": model,
              "temperature": 0,
              "messages": [
                  {"role": "system", "content": "You are a financial expert."},
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
              return f"[ERROR] {str(e)}"

7. Main RAG Evaluation Function

   .. code-block:: python

      def evaluate_with_rag(questions):
          """Process CFA questions with RAG and DeepSeek"""
          print(f"Processing {len(questions)} Kaplan Schweser CFA questions")
          
          # Load RAG components
          index, chunks = load_index_and_chunks()
          
          print(f"{'Question':<50} | {'True Answer':<11} | {'Model Answer':<12} | {'Correct':<7}")
          print("-" * 85)
          
          correct_count = 0
          
          # Process each question
          for i, question_data in enumerate(tqdm(questions, desc="Processing with RAG")):
              question = question_data["question"]
              choices = question_data["choices"]
              true_answer = question_data["answer"]
              
              # Get relevant chunks using RAG
              relevant_chunks = retrieve_chunks(question, index, chunks)
              context = "\n\n".join(relevant_chunks)
              
              # Create prompt with RAG context
              prompt = make_prompt(question, choices, context=context)
              
              # Get answer from DeepSeek with RAG
              rag_answer = ask_deepseek(prompt)
              
              # Check if answer is correct
              is_correct = rag_answer.strip() == true_answer.strip()
              if is_correct:
                  correct_count += 1
                  
              # Print result in table format
              truncated_question = question[:45] + "..." if len(question) > 45 else question
              print(f"{truncated_question:<50} | {true_answer:<11} | {rag_answer:<12} | {'✓' if is_correct else '✗'}")
              
              # Print detailed analysis for first question
              if i == 0:
                  print("\nDetailed Analysis of First Question:")
                  print(f"Question: {question}")
                  print("Choices:")
                  for letter, text in choices.items():
                      print(f"  {letter}. {text}")
                  print(f"Correct Answer: {true_answer}")
                  print(f"Model's Answer: {rag_answer}")
                  print(f"Explanation: {question_data.get('explanation', 'No explanation provided')}")
          
          # Print summary statistics
          accuracy = (correct_count / len(questions)) * 100
          print(f"Total Questions: {len(questions)}")
          print(f"Correct Answers: {correct_count}")
          print(f"Accuracy: {accuracy:.2f}%")
          print(f"\nQuestions sourced from Kaplan Schweser CFA exam materials")

8. Main Execution

   .. code-block:: python

      if __name__ == "__main__":
          # Evaluate using Kaplan Schweser questions
          evaluate_with_rag(kaplan_schweser_questions)

Running the Tutorial
--------------------

1. Create a config.json file with your API keys
2. Ensure you have the RAG resources (FAISS index, chunks file)
3. Save code as ``benchmark_deepseek_rag.py``
4. Run with ``python benchmark_deepseek_rag.py``

Example Output
-------------

.. code-block:: text

    Loading index from ./RAG/CFA_RAG/faiss_index.idx
    Loading chunks from ./RAG/CFA_RAG/chunks.pkl
    Loaded index with 4520 vectors and 4520 chunks
    Processing 4 Kaplan Schweser CFA questions
    Processing with RAG: 100%|██████████| 4/4 [00:20<00:00, 5.12s/it]
    
    === CFA Knowledge Evaluation Results ===
    Question                                          | True Answer | Model Answer | Correct
    ---------------------------------------------------------------------------------
    If the market yield does not change, the pric... | A           | A            | ✓
    
    Detailed Analysis of First Question:
    Question: If the market yield does not change, the price of a Treasury bill:
    Choices:
      A. Will increase as the bill approaches maturity.
      B. Will decrease as the bill approaches maturity.
      C. Stay the same as the bill approaches maturity.
    Correct Answer: A
    Model's Answer: A
    Explanation: Treasury bills are discount instruments. As the bond approaches maturity, the price would increase.
    
    === Summary ===
    Total Questions: 4
    Correct Answers: 3
    Accuracy: 75.00%
    
    Questions sourced from Kaplan Schweser CFA exam materials

Notes
-----

- **RAG (Retrieval-Augmented Generation)** enhances model answers by providing relevant context from a knowledge base
- **Embeddings** are vector representations of text that capture semantic meaning
- **FAISS** is a library for efficient similarity search of dense vectors
- **Vector similarity** allows finding the most relevant chunks for each question
- **Context window** in RAG provides the model with retrieved information, reducing hallucinations

Additional Resources
-------------------
- `DeepSeek API Documentation <https://api.deepseek.com/docs>`_
- `FAISS Library <https://github.com/facebookresearch/faiss>`_
- `OpenAI Embeddings <https://platform.openai.com/docs/guides/embeddings>`_
- `Kaplan Schweser CFA Level 1 Mock Exam Sample Questions <https://www.schweser.com/cfa/blog/how-to-pass-the-cfa-exam/cfa-exam-sample-questions?srsltid=AfmBOopIhyo8K27Hz7W4r6s7eoewIrvkx-2oxNt9impeiNpxS-yRdetf>`_ 