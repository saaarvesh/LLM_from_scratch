# 🧠 Building GPT-2 (124M) from Scratch — A Journey Into LLMs

Hi 👋  

This repo exists because I wanted to *really* understand every nut and bolt of how **large language models (LLMs)** like GPT work.  
So I built the **complete GPT-2 architecture (124M parameters)** entirely **from scratch in PyTorch**, layer by layer — no shortcuts, no libraries, just me and `Pytorch`.

And yeah... I trained it **on CPU** 😅.  
No GPU, no TPU — just pure curiosity and stubbornness.  
Learning shouldn’t stop just because hardware does 🚀  

Initial training was done on a small storybook dataset called **“Verdic”**, just to get things working end-to-end.

After training and fine-tuning, I also added **Top-k sampling** and **Temperature scaling** during text generation to improve diversity and control randomness.

---

## ⚙️ Overview

This project started as a learning experiment and slowly evolved into a complete, working GPT-2 replica —  
from **tokenizer → embeddings → transformer → language model head → training → Sampling Improvements: Top-k & Temperature Scaling → fine-tuning → evaluation**.

Then I didn’t stop there 😄  
After verifying my model architecture, I:
- Loaded **OpenAI GPT-2 pretrained weights** into my implementation  
- Did **classification fine-tuning** (spam vs non-spam)  
- Tried **instruction fine-tuning** (Alpaca-style)  
- And finally **evaluated my model** with **ollama** using **Llama-3-2B** as the *external judge*.  

---

## 📂 Folder Structure

Here’s the full repo layout for clarity:

```
Root
├── README.md                # this file
│
├── Code_base/
│   ├── 1.Tokenizer/                   # tokenizer experiments (BPE, tokenization notes)
│   ├── 2.Dataloader_and_embeddings/   # batching & embeddings pipeline
│   ├── 3.E2e_data_processing/         # dataset cleaning and preprocessing notebooks
│   ├── 4.Attention/                   # toy and trainable attention notebooks (step-by-step)
│   ├── 5.GPT Architechture/           # GPT model implementation, training/evaluation notebooks, plots, checkpoints
│   └── 6.Finetuning/                  # finetuning notebooks and scripts (classification & instruction tuning)
│
└── Dataset/
    ├── the-verdict.txt                # main training text (storybook)
    ├── spam_dataset.csv               # spam/non-spam classification data
    ├── instruction_data.json          # Alpaca-style instruction dataset
```

---

## 🏗️ Model & Training Details

| Component | Description |
|------------|-------------|
| **Framework** | PyTorch |
| **Model** | GPT-2 Small (124M parameters) |
| **Layers** | 12 Transformer blocks, 12 attention heads, 768 hidden dim |
| **Positional Encoding** | Learned embeddings |
| **Activation** | GELU |
| **Optimizer** | AdamW |
| **Loss Function** | Cross-Entropy |
| **Context Length** | 256 tokens |
| **Vocab Size** | ~50k (GPT-2 BPE compatible) |
| **Hardware** | CPU only (no GPU) |
| **Dataset** | “Verdic” storybook text |

Even with limited compute, the model successfully ran through the full training pipeline:
- tokenization  
- batching  
- forward + backward pass  
- weight updates  
- and basic text generation 🎉

---

## 🔧 Fine-Tuning Experiments

### 🧠 1. Loading Pretrained GPT-2 Weights
After validating my own GPT-2 architecture, I loaded **OpenAI’s official GPT-2 (124M)** weights into my implementation.  
Layer shapes matched perfectly — proving my model replicated the real GPT-2 spec accurately.

---

### 📬 2. Spam / Non-Spam Classification
Once the pretrained weights were loaded, I fine-tuned GPT-2 for a **binary text classification** task.  

**Approach:**
- Froze **all transformer layers** except:
  - Final Transformer block  
  - Final LayerNorm  
  - Classification head (modified MLP → 2 logits)
- Loss: Binary Cross-Entropy  
- Goal: Teach GPT-2 to act as a classifier with minimal fine-tuning.

This was my first test of *repurposing a generative LLM for classification.*

---

### 💬 3. Instruction Fine-Tuning (Alpaca-Style)
Then came the fun part 😄  

I formatted custom **instruction-following data** (in the style of Alpaca):

```json
{
  "instruction": "Summarize the paragraph.",
  "input": "Once upon a time...",
  "output": "It tells a story about..."
}
```

Then I fine-tuned the pretrained GPT-2 on this **manually curated dataset** to make it more conversational and capable of following prompts.

**Goal:** Make my small GPT-2 “talk” like an instruction-tuned model.

---

## 🧪 Evaluation

After instruction fine-tuning, I wanted to see **how well my GPT-2-from-scratch model followed instructions** compared to the expected dataset outputs.

For this, I used **OLLa (Open LLM Arena)** with **LLaMA 3.2B** acting as an *LLM judge* — it scored my model’s responses on relevance, correctness, and clarity.

Below are two examples from the evaluation:

---

### 🧾 Example 1

**Dataset response:**
> The car is as fast as lightning.  

**Model response:**
> The car is as fast as a bullet.  

**Judge (LLaMA 3.2B) score:**
> **85 / 100**

**Reasoning (summarized):**
- Correctly uses a simile structure (“as fast as …”).  
- Comparison is relevant — bullets imply high velocity.  
- Slightly less vivid than “lightning,” but still effective.  

✅ *A strong, contextually appropriate response showing the model can produce figurative language accurately.*

---

### 🧾 Example 2

**Dataset response:**
> The type of cloud typically associated with thunderstorms is cumulonimbus.  

**Model response:**
> A thunderstorm is a type of cloud that typically forms in the atmosphere over a region of high pressure...  

**Judge (LLaMA 3.2B) score:**
> **20 / 100**

**Reasoning (summarized):**
- Did **not** answer the actual question.  
- Provided irrelevant or inaccurate meteorological info.  
- Failed to identify “cumulonimbus” as the correct answer.  

❌ *Shows where factual grounding still needs improvement — the model drifted off-topic and generated plausible but incorrect text.*

---

These evaluations highlight that while my instruction-tuned GPT-2 can **handle basic analogy and phrasing tasks well**, it still **struggles with factual recall and question precision** — common challenges for small LLMs without large-scale pretraining.


## 💡 Key Learnings

- Built GPT-2 architecture entirely from scratch — no copying Hugging Face code.  
- Understood attention, layer norm, and feed-forward layers *deeply*.  
- Learned why **cross-entropy**, **masking**, and **residual connections** are critical.  
- Saw how easily pretrained models can be **repurposed** with small fine-tuning.  
- And most importantly — that even **on CPU**, you can still explore LLMs end-to-end 💪  

---

## 🛠️ Future Plans

- Add **Rotary embeddings (RoPE)** and compare to learned positional encodings  
- Implement **parameter-efficient fine-tuning (LoRA)**  
- Try **distributed GPU training** for larger experiments   
- Experiment with **text summarization fine-tuning**

---

## 🤝 Acknowledgments
- [Sebastian Raschka Book](https://github.com/rasbt/LLMs-from-scratch)
- 🧾 [OpenAI GPT-2 paper (2019)](https://openai.com/research/language-unsupervised)
- [OpenAI GPT-2 Open Weights](https://openaipublic.blob.core.windows.net/gpt-2/models) 
- 🧑‍🏫 [Vizuara’s *“LLM from scratch”* Playlist](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)  
- 🤗 Hugging Face Transformers (for reference implementations)  

---

## 🚀 Final Thought

This project isn’t about beating GPT-2 — it’s about **understanding GPT-2**.  
Every line of code here was written, debugged, and tested manually to *see how transformers think*.  

If you’re someone who learns best by building things from scratch —  
you’ll feel right at home here ❤️  

---

⭐ **If you find this repo interesting, give it a star!**  
Learning should always be open-source 💪
