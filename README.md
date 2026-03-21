# Playing with GRPO: Teaching Mistral-7B to “Think”

I wanted to experiment with **GRPO (Group Relative Policy Optimization)** to see whether I could induce _explicit reasoning behavior_, similar in spirit to DeepSeek-R1, on a smaller open model.

The setup is intentionally simple:

- Start with **Mistral-7B Instruct v0.3**
- Warm it up with **SFT on GSM8K**
- Apply **GRPO** to encourage structured reasoning via XML-style `<reasoning>` traces
- Measure whether this _actually improves math accuracy_, not just verbosity

---

## Experiment Setup

### Model & Dataset

- **Model:** [unsloth/mistral-7b-instruct-v0.3](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3)
- **Dataset:** [GSM8K](https://huggingface.co/datasets/gsm8k)

To keep evaluation clean and avoid leakage, the GSM8K _training_ split was subdivided as follows:

| Split Source | Usage         | Count |
| ------------ | ------------- | ----- |
| Train        | SFT           | 4,000 |
| Train        | RL (GRPO)     | 2,900 |
| Train        | Internal Eval | 500   |
| Test         | Final Eval    | 1,319 |

---

### Hardware

Different stages had very different memory profiles:

- **Zero-shot baseline & SFT**
  Kaggle - _2× T4 (15GB VRAM each)_

- **RL (GRPO) & Final Evaluation**
  LightningAI Studio - _1× L4 (24GB VRAM)_
  vLLM was used during evaluation for faster generation.

---

## Results

### 1. GSM8K Test Set (Unseen, Final Numbers)

This is the metric that actually matters.

| Stage         | Accuracy   | Notes                       |
| ------------- | ---------- | --------------------------- |
| Zero-Shot     | 41.17%     | Mistral-7B Instruct v0.3    |
| SFT           | 44.50%     | 4k supervised examples      |
| **RL (GRPO)** | **52.46%** | Structured reasoning via RL |

**GRPO yields a +8% absolute gain over SFT on the unseen test set.**

---

### 2. Internal Train-Eval Split (500 examples)

| Stage     | Accuracy |
| --------- | -------- |
| Zero-Shot | 41.20%   |
| SFT       | 58.00%   |
| RL        | 45.40%\* |

- This apparent drop is **not a real performance regression**. See below.

---

## Why Does RL Look Worse on the Train-Eval Split?

At first glance, RL appears to _hurt_ performance on the 500-example internal eval split (58.00% → 45.40%).
This initially raised concerns.

After auditing the code paths, the cause was identified clearly:

> **The models were not evaluated with the same correctness function.**

### What actually went wrong

Each notebook used a **slightly different answer-extraction + comparison rule**:

#### ZS / SFT notebooks

- Extracted the _last number as a string_
- Used strict string equality
- Very lenient toward formatting
- Did **not** expect structured output

#### RL notebook

- Expected answers inside `<answer>...</answer>` XML tags
- Used **numeric (float) comparison**
- Penalized:

  - missing tags
  - truncated generations
  - correct math with malformed structure

As a result, the RL model was evaluated under a **strictly harsher criterion** during internal evaluation.

This makes the RL score on the train-eval split **not directly comparable** to the ZS and SFT scores.

---

### How this was fixed

For the **final GSM8K test evaluation**, a **single unified extraction function** was used for _all_ models:

1. Prefer `<answer>...</answer>` (RL-style)
2. Fall back to `####` (GSM8K / SFT-style)
3. Fall back to the last numeric token
4. Compare numerically with tolerance

Once evaluated consistently, the RL model clearly outperforms both baselines.

---

### Key takeaway

The lower RL score on the train-eval split is an **evaluation artifact**, not a modeling failure.

> **This experiment reinforced a critical lesson:**
> In RL for language models, _evaluation consistency matters as much as the training algorithm itself._

---

## Experiment Flow

The project follows a standard alignment pipeline:

1. **`sft-gsm8k.ipynb` - Supervised Fine-Tuning**

   - Teaches the base model GSM8K-style answering
   - Establishes a strong baseline

2. **`rl-gsm8k.ipynb` - Reinforcement Learning (GRPO)**

   - Encourages explicit reasoning traces using XML tags
   - Rewards correctness of the final numeric answer
   - Trades some formatting stability for better reasoning

3. **`eval-gsm8k.ipynb` - Final Evaluation**

   - Uses vLLM for fast generation
   - Applies a unified, format-agnostic correctness function

---

## Final Thoughts

This was not about chasing SOTA numbers.
It was about understanding **how GRPO actually behaves** in practice:

- RL _does_ improve reasoning generalization
- Output format matters more than expected
- Evaluation bugs can completely invert conclusions if left unchecked

Overall, GRPO proved to be a **practical and effective way** to push a 7B-scale model toward better reasoning, when used carefully.
