"""
Fine-tune google/muril-base-cased for binary Spam / Ham classification.

Reads train/test CSVs produced by prepare_dataset.py and saves the
fine-tuned model + tokenizer to fine_tuned_model/.

Usage:
    python fine_tune.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from config import (
    DATA_DIR,
    FINE_TUNED_MODEL_DIR,
    TOKENIZER_NAME,
    FINETUNE_EPOCHS,
    FINETUNE_BATCH_SIZE,
    FINETUNE_LEARNING_RATE,
    FINETUNE_MAX_LENGTH,
    CLASSIFIER_LABELS,
)

# ──────────────────────────── helpers

def compute_metrics(eval_pred):
    """Compute accuracy and F1 for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def load_data():
    """Load train/test CSVs into HuggingFace Datasets."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Ensure text column is string
    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
    return train_ds, test_ds


# ──────────────────────────── main

def main():
    print("=" * 60)
    print("  MuRIL Fine-Tuning for Spam Classification")
    print("=" * 60)

    # 1. Load tokenizer
    print(f"\n[*] Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # 2. Load data
    print("[*] Loading datasets ...")
    train_ds, test_ds = load_data()
    print(f"    Train: {len(train_ds)} | Test: {len(test_ds)}")

    # 3. Tokenize
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=FINETUNE_MAX_LENGTH,
        )

    print("[*] Tokenizing ...")
    train_ds = train_ds.map(tokenize_fn, batched=True, batch_size=256)
    test_ds = test_ds.map(tokenize_fn, batched=True, batch_size=256)

    # Set format for PyTorch
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 4. Load model
    print(f"[*] Loading model: {TOKENIZER_NAME} (num_labels={len(CLASSIFIER_LABELS)})")
    model = AutoModelForSequenceClassification.from_pretrained(
        TOKENIZER_NAME,
        num_labels=len(CLASSIFIER_LABELS),
    )

    # 5. Training arguments (CPU-optimized)
    os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=FINE_TUNED_MODEL_DIR,
        num_train_epochs=FINETUNE_EPOCHS,
        per_device_train_batch_size=FINETUNE_BATCH_SIZE,
        per_device_eval_batch_size=FINETUNE_BATCH_SIZE,
        learning_rate=FINETUNE_LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",          # no wandb / tensorboard
        fp16=False,                # CPU — no fp16
        dataloader_num_workers=0,  # Windows-safe
        use_cpu=True,
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("\n[*] Starting training ...\n")
    trainer.train()

    # 7. Evaluate
    print("\n[*] Evaluating on test set ...")
    metrics = trainer.evaluate()
    print(f"    Accuracy : {metrics['eval_accuracy']:.4f}")
    print(f"    F1 Score : {metrics['eval_f1']:.4f}")

    # 8. Save
    print(f"\n[*] Saving model to {FINE_TUNED_MODEL_DIR}")
    trainer.save_model(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)

    print("[✓] Fine-tuning complete!")


if __name__ == "__main__":
    main()
