# Natural-Language-Final-Project
Author: Iván Montalvo

Format: Jupyter Notebook (`Final_Assignment_Ivan_Montalvo.ipynb`)

Companion file: `results.json`

## What this is
A Jupyter notebook that fine-tunes a small open-weight language model on a dataset of dictionary entries for the Muysca language, then lets you ask it questions about Muysca vocabulary and get a generated answer.

## The dataset (`results.json`)
The JSON file containing ~2000 entries. Each entry has these fields:

- words: the headword
- urls: a source URL (`muysca.cubun.org`, the online Muysca language repository)
- text: the Spanish-language definition of the word
- Tokens: tokens found in the definition.¿
- NER: test entry for an attempt to get the NER from the words

IMPORTANT: The JSON file needs to be uploaded into the working directory before running the rest of the code.

## How the notebook is structured

1. Setup

- Install `unsloth`
  - the install path differs depending on whether a COLAB_ environment variable is detected. A plain `pip install unsloth` otherwise, or a more manual install of `unsloth` dependencies if running on Colab (see below).

- Install `datasets` and pins `transformers==4.51.3`.
- Import `torch` and `datasets.load_dataset`.

  * I couldn't mount Google Drive for `results.json` without a paid plan (due to the model's size), so the file has to be uploaded manually from a shared Drive link instead.

2. Model loading

Load `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`  via `FastLanguageModel.from_pretrained`, with `max_seq_length = 1024`.

3. LoRA configuration

Wrap the model with `FastLanguageModel.get_peft_model` for parameter-efficient fine-tuning, targeting the attention projections and MLP projections, `lora_alpha=16`, no dropout, gradient checkpointing enabled.

4. Dataset preparation

Load `results.json` with `datasets.load_dataset`, then convert it into a ShareGPT-style conversational format using Unsloth's `to_sharegpt`, mapping the text field into a templated prompt, with `conversation_extension=3`. It then standardizes the format with `standardize_sharegpt`.

5. Chat template

Define a Llama-3-style chat template (`<|begin_of_text|>`, `<|start_header_id|>` tags for system/user/assistant turns) and applies it to the dataset with `apply_chat_template`, using the default system message: <i>"You are a helpful assistant that gives concise and short answers about languages."</i>

6. Training

I set up `trl.SFTTrainer` with the following `TrainingArguments`: 
- batch size: 2
- gradient accumulation: 4
- warmup steps: 5
- max steps: 50
- epoch: 1
- learning rate: 1e-4
- optimizer: adamw_8bit
- weight decay: 0.01
- schedule: linear LR
- seed: 3407
- precision: mixed (fp16 or bf16 depending on hardware support)

Then call `trainer.train()`.

7. Inference test

Run the fine-tuned model on a sample message ("Hi, How are you?") using `TextStreamer` to stream the generated response token-by-token.

8. Saving

Save the LoRA adapter weights and tokenizer locally to a folder called `lora_model`.

10. Interactive question cell

The final cell lets you edit a target string to ask the model a question and stream back its answer. The example included in the notebook is "¿Que significa huihichua en Muysca?" ("What does huihichua mean in Muysca?").

## How to run it

1. Open the notebook (preferably on Google Colab).
2. Upload `results.json` into the working directory. Make sure to save file at `../content/results.json`.
3. Run the cells top to bottom
4. After training finishes, run the inference cells, and edit the target variable in the last cell to ask your own question.

## Requirements

- GPU runtime (the model loading and training cells use 4-bit quantization and CUDA)
- Python packages: `torch`, `datasets`, `transformers==4.51.3`, `unsloth`, `trl`
- Unsloth dependencies (`bitsandbytes`, `accelerate`, `xformers`, `peft`, `triton`, `cut_cross_entropy`, `unsloth_zoo`, `sentencepiece`, `protobuf`, `huggingface_hub`, `hf_transfer`)
