# finetuning_llm_for_narutodata
Done as a part of assignment of internship application
# 🌀 Naruto Personality LLM — Fine-tuning GPT-2 using QLoRA

This project focuses on building a lightweight language model that **responds like Naruto Uzumaki**, the beloved character from the Naruto anime. Using **subtitle data** from episodes, we extract Naruto’s dialogue, thinking patterns, personality traits, and fine-tune a quantized GPT-2 model using **QLoRA (Quantized Low-Rank Adaptation)**.

---

## 🧠 Goal

> To create a **personality-infused LLM** that behaves like Naruto Uzumaki — optimistic, loyal, determined — in conversation with users. The model will reflect his philosophies, speech patterns, and character traits by learning from his **actions and dialogue** in the anime.

---

## 🛠️ Key Components

### 1. Subtitle Data Extraction

We start by parsing `.srt`, `.ass`, and `.txt` subtitle files from the anime. We extract lines spoken by Naruto and other characters, clean them, and store them in a structured form.

**Tool used**: `re`, `glob`, Python text preprocessing

---

### 2. Summarization + Character Role Extraction

Each episode transcript is analyzed using GPT (via OpenAI or Gemini) to extract:
- `theme`: the main theme of the episode (friendship, dreams, etc.)
- `summary`: plot-level summary
- `naruto_role_personality_thinking`: Naruto’s mindset, values, and notable quotes

This generates a file like:

```json
[
  {
    "theme": "dreams, determination",
    "summary": "Naruto and Sasuke fight Haku. Naruto emphasizes never giving up.",
    "naruto_role_personality_thinking": "Naruto says, 'I can't die here... I have a dream to fulfill.'"
  }
]
3. Instruction-Tuned Data Creation (finetune.jsonl)
Using the structured data, we convert it to instruction-style prompt–completion pairs:

jsonl
Copy
Edit
{"prompt": "Naruto Uzumaki is optimistic...\nA friend says: “Nobody believes in me.”\nNaruto:\n", 
 "completion": " You're wrong! I believe in you, just like someone believed in me once. You can change everything!"}
This format allows GPT-2 to learn how Naruto responds based on the intent and tone of the user’s input.

4. Fine-tuning with QLoRA
We apply QLoRA to fine-tune a 4-bit quantized version of gpt2-medium:

🚀 8-bit and 4-bit quantization reduces GPU memory usage

🔧 LoRA applies low-rank updates instead of full weight training

✅ Efficient: can train even on small Colab GPUs

Libraries:

transformers

peft (for LoRA)

bitsandbytes (for quantization)

python
Copy
Edit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
Trainer is configured with:

load_in_4bit = True

Gradient Accumulation

LR Scheduler

max_seq_length = 512

5. Inference
Once trained, the model can generate Naruto-style responses:

python
Copy
Edit
naruto_reply("I feel like giving up.")
# 🌀 Output: “You can't give up! That's not how we do things! Even if it's hard, believe in yourself like I did.”
🧪 Example Interaction
python
Copy
Edit
naruto_reply("Everyone hates me and I can't take it anymore.")
vbnet
Copy
Edit
Naruto: "You know... I used to feel the same way. But I never gave up. I trained, I fell, I stood up. You have something special too — believe it!"
⚙️ Dependencies
bash
Copy
Edit
pip install transformers accelerate peft bitsandbytes datasets
Or use Colab with GPU.
#Improvemewnts:
1.Use larger llms like mistral 7b,etc..
2. more synthetic data generation dealing for vaarious scenearios accordingly and use it for the finetuning.
