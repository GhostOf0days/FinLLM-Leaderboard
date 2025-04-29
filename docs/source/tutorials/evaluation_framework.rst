Evaluate Deepseek using Framework
==============================================

.. contents:: Table of Contents
   :local:

Overview
--------
This evaluation framework is a guided workflow designed to help you benchmark your language models on various tasks using Google Colab. 

This framework:

- Installs all required libraries, downloads model checkpoints, and configures necessary paths.
- Executes evaluation scripts from your GitHub repository to measure model performance on defined tasks.
- Stores performance outputs (such as accuracy or F1 scores) in text files for later review.
- Easily replace placeholders (e.g., your repository link, model URLs) to tailor the evaluation to your needs.

Prerequisites
-------------

Before you begin, ensure you have:
- A **Google Colab** account with GPU access (preferably an A100 or similar).
- A valid **Hugging Face Token** for accessing models that require authentication.
- Your own **GitHub repository** containing the evaluation scripts.
- Any necessary model checkpoints or configuration files (stored on Google Drive or locally).

Environment Setup
-----------------
In this section, you'll install the required packages, clone your repository, and configure the environment.

Installation and Repository Cloning
+++++++++++++++++++++++++++++++++++++
Run the following commands in a Colab cell to upgrade PyTorch, clone your repository, and install the required libraries. Replace `<YOUR_GITHUB_REPO_LINK>` with your repository URL.

.. code-block:: bash

   # Upgrade torch and clone your GitHub repository
   !pip install -U torch
   !git clone -b working <YOUR_GITHUB_REPO_LINK> --recursive
   %cd PIXIU
   !pip install -r requirements.txt


Installing Additional Dependencies
+++++++++++++++++++++++++++++++++++++
Navigate to the evaluation folder and install any extra packages required by your tasks:

.. code-block:: bash

   %cd /content/PIXIU/src/financial-evaluation
   !pip install -e .[multilingual]
   !pip install bert_score gdown vllm==0.5.4 torch==2.4.0 torchvision==0.19 peft lm-eval google-generativeai

Configuration and Checkpoint Setup
------------------------------------
Many evaluation tasks require specific model checkpoints. For instance, the BART score uses a dedicated checkpoint. Use the snippet below to mount Google Drive and either copy or download the checkpoint file.

.. code-block:: bash

   from google.colab import drive
   import os
   import gdown


   # Mount Google Drive to access the checkpoint
   drive.mount('/content/drive')

   source_path = "/content/drive/My Drive/bart_score.pth"
   destination_path = "/content/PIXIU/src/metrics/BARTScore/bart_score.pth"

   if os.path.exists(source_path) and not os.path.exists(destination_path):
       !cp "{source_path}" "{destination_path}"
       print("Checkpoint copied from Google Drive.")
   else:
       file_id = 'FILE_ID_HERE' # Replace with your file ID here
       url = f'https://drive.google.com/uc?id={file_id}'
       gdown.download(url, destination_path, quiet=False)
       print("Checkpoint downloaded from remote URL.")

Setting PYTHONPATH and Authentication
----------------------------------------
Configure the PYTHONPATH so that Python can locate evaluation modules, then log in to Hugging Face.  
**Make sure you have access to your Hugging Face Token, since some datasets require requesting access.**

.. code-block:: bash

   %cd /content/PIXIU/src
   %cd /content

   import os
   os.environ['PYTHONPATH'] += ":/content/PIXIU/src/metrics/BARTScore/"
   !echo $PYTHONPATH

   # Log in to Hugging Face: Replace "YOUR_HUGGINGFACE_TOKEN" with your actual token.
   # Make sure you have access to your Hugging Face Token, since some datasets require requesting access.
   from huggingface_hub import login
   login(token="YOUR_HUGGINGFACE_TOKEN")

Running Evaluation Tasks
------------------------
Define your evaluation tasks and set the model parameters. Update the transformer URLs with your model's pretrained and tokenizer URLs.

.. code-block:: markdown

   **Tasks Names Defined Below**

   - **NER:** flare_ner
   - **FINER-ORD:** flare_finer_ord
   - **FinRED:** flare_finred
   - **SC:** flare_causal20_sc
   - **CD:** flare_cd
   - **FNXL:** flare_fnxl
   - **FSRL:** flare_fsrl
   - **FPB:** flare_fpb
   - **FiQA-SA:** flare_fiqasa
   - **TSA:** flare_tsa
   - **Headlines:** flare_headlines
   - **FOMC:** flare_fomc
   - **FinArg-ACC:** flare_finarg_ecc_auc
   - **FinArg-ARC:** flare_finarg_ecc_arc
   - **MultiFin:** flare_multifin_en
   - **MA:** flare_ma
   - **MLESG:** flare_mlesg
   - **FinQA:** flare_finqa
   - **TATQA:** flare_tatqa
   - **Regulations:** (No specific task name provided for this)
   - **ConvFinQA:** flare_convfinqa
   - **EDTSUM:** flare_edtsum
   - **ECTSUM:** flare_ectsum
   - **BigData22:** flare_sm_bigdata
   - **ACL18:** flare_sm_acl
   - **CIKM18:** flare_sm_cikm
   - **German:** flare_german
   - **Australian:** flare_australian
   - **LendingClub:** flare_cra_lendingclub
   - **ccf:** flare_cra_ccf
   - **ccfraud:** flare_cra_ccfraud
   - **polish:** flare_cra_polish
   - **taiwan:** flare_cra_taiwan
   - **portoseguro:** flare_cra_portoseguro
   - **travelinsurance:** flare_cra_travelinsurace
   - **ES_FinanceES:** flare_es_financees
   - **ES_Multifin:** flare_es_multifin
   - **ES_EFP:** flare_es_efp
   - **ES_EFPA:** flare_es_efpa
   - **ES_FNS:** flare_es_fns
   - **ES_TSA:** flare_es_tsa

.. code-block:: bash

   tasks_list = [
       "flare_ner",
       "flare_finer_ord",
       "flare_finred",
       "flare_causal20_sc",
       ...
       # Add or remove tasks as needed.
   ]

   pretrained = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" # Replace with your pretrained transformer URL here
   tokenizer = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" # Replace with your tokenizer transformer URL here
   max_gen_toks = 512 # Input necessary tokens needed
   batch_size = 20000
   num_fewshot = 0
   results_dir = "/content/results"
   model_type = "hf-causal-vllm"
   model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" #Replace with your model name

   .. note::

   Here we set the evaluation parameters by defining the transformer model, tokenizer, token generation limit, batch size, and output directory for running the evaluation.

   # Create a directory for your results
   import os
   os.makedirs(f"{results_dir}/{model_name}", exist_ok=True)

   # Run the evaluation for each task and store the outputs
   for task in tasks_list:
       output_file_path = f"{results_dir}/{model_name}/{task}_results.txt"
       print(f"Running task: {task}")
       print(f"Saving output to: {output_file_path}\n")

       !python PIXIU/src/eval.py \
           --model $model_type \
           --model_args "pretrained=$pretrained,tokenizer=$tokenizer,trust_remote_code=True,use_fast=False,max_gen_toks=$max_gen_toks" \
           --tasks $task \
           --batch_size $batch_size \
           --num_fewshot $num_fewshot \
           --output_base_path $results_dir \
           > $output_file_path

.. note:: 
   
   This code block creates a results directory for the model and runs each evaluation task, saving the outputs to corresponding files.

Example Output Snippet
-----------------------
After running an evaluation task, check the output file in your results folder. For example, the following is a snippet from the output file of the "flare_ner" task:

.. code-block:: text

   Running task: flare_ner
   Saving output to: /content/results/<YOUR_MODEL_NAME>/flare_ner_results.txt

   2025-04-08 21:24:36.097889: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on...
   2025-04-08 21:24:36.115047: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] ...
   WARNING: All log messages before absl::InitializeLog() are written to STDERR
   ...
   Loading safetensors checkpoint shards: 100% 2/2 [00:04<00:00,  2.45s/it]
   0it [00:00, ?it/s]

   ...
   Final Results (from flare_ner_results.txt):
   --------
   hf-causal-vllm (pretrained=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,tokenizer=deepseek-ai/DeepSeek-R1-Distill-Llama-8B,trust_remote_code=True,use_fast=False,max_gen_toks=128), limit: None, provide_description: False, num_fewshot: 0, batch_size: 20000
   |  Task   |Version| Metric  |Value |   |Stderr|
   |---------|------:|---------|-----:|---|------|
   |flare_ner|      1|entity_f1|0.0127|   |      |

Running the Tutorial
--------------------
To run this evaluation framework: 

- **Replace placeholders:** Update `<YOUR_GITHUB_REPO_LINK>`, `<INSERT_PRETRAINED_TRANSFORMER_URL_HERE>`, `<INSERT_TOKENIZER_TRANSFORMER_URL_HERE>`, `<YOUR_MODEL_NAME>`, and `"YOUR_HUGGINGFACE_TOKEN"` with your actual values.
- **GPU configuration:** Ensure that your Google Colab runtime is set to use a GPU (e.g., A100).
- **Execute cells sequentially:** Copy the code blocks into separate Colab cells and run them in sequence. The output files with performance metrics will be saved in the specified results directory.

Key Takeaways
-------------
- This framework guides you through setting up, executing, and monitoring the evaluation process.
- You can add your repository links, model URLs, and tasks without altering the overall structure.
- The results file provides key performance metrics (e.g., F1 score) to help you assess and improve your models.
- Remember to review the output logs for any warnings or errors, as this information is crucial for troubleshooting and optimizing your evaluation process.
