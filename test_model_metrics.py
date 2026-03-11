import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classifier import ONNXClassifier
import pandas as pd
import os

def evaluate_model(csv_path=None):
    """
    Evaluates the fine-tuned MuRIL hybrid classifier.
    If csv_path is provided, it loads data from there (needs 'text' and 'label' columns).
    Otherwise, it uses a hardcoded representative sample set.
    """
    classifier = ONNXClassifier()
    
    try:
        classifier.load()
    except FileNotFoundError as e:
        print(f"[!] {e}")
        sys.exit(1)
        
    if csv_path and os.path.exists(csv_path):
        print(f"[*] Loading test data from: {csv_path}")
        df = pd.read_csv(csv_path)
        # Ensure correct column names
        if 'text' not in df.columns or 'label' not in df.columns:
            print("[!] CSV must contain 'text' and 'label' columns.")
            sys.exit(1)
        test_data = df.to_dict('records')
    else:
        print("[*] No test CSV found. Using built-in hybrid sample dataset...")
        test_data = [
            {"text": "Congratulations! You've won a free prize. Call now to claim!", "label": "Spam"},
            {"text": "Hey, are you coming to the meeting tomorrow?", "label": "Ham"},
            {"text": "Sir, bank loan offer irukku, 2% interest la kedaikum.", "label": "Spam"},
            {"text": "Naan late varuvena, 15 minutes la varuven.", "label": "Ham"},
            {"text": "Sir our number is OTP sir, OTP number, so we can enter the OTP and enter the address.", "label": "Spam"},
            {"text": "Bro, cricket match paakka pogalama?", "label": "Ham"},
            {"text": "Sir claim pannunga sir, ungalukku special offer irukku.", "label": "Spam"},
            {"text": "Your bank account has been compromised. Verify immediately!", "label": "Spam"},
            {"text": "Can you please send me the presentation file?", "label": "Ham"},
            {"text": "Enakku inniki konjam udambu mudiyala, office vara maaten.", "label": "Ham"},
            {"text": "Hello madam, neenga select aagitinga 1 lakh prize ku. Details anupunga.", "label": "Spam"},
            {"text": "Aadhaar card suspend agirchu, update panna indha link click pannunga.", "label": "Spam"},
            {"text": "Dei evening yenga polam?", "label": "Ham"},
            {"text": "Amazon customer support calling. Your package is delayed.", "label": "Ham"},
            {"text": "Urgent! Your credit card is blocked due to suspicious activity. Call this number.", "label": "Spam"},
            {"text": "Hi sir, can I get your email ID to send the invoice?", "label": "Ham"},
        ]

    y_true = []
    y_pred = []
    
    print("=" * 70)
    print("Evaluating Fine-Tuned Model (MuRIL + Keyword Boost)".center(70))
    print("=" * 70)
    
    print(f"[*] Processing {len(test_data)} samples...")
    
    for item in test_data:
        text = str(item["text"])
        # Format labels safely (in case dataset has 0/1 or spam/ham lowercase)
        true_label = str(item["label"]).strip().capitalize()
        if true_label not in ["Spam", "Ham"]:
            true_label = "Spam" if true_label in ["1", "true"] else "Ham"
            
        y_true.append(1 if true_label == "Spam" else 0)
        
        # Predict using our hybrid classifier
        result = classifier.classify(text)
        pred_label = result["label"]
        y_pred.append(1 if pred_label == "Spam" else 0)
        
        # Print misclassifications to understand model weaknesses
        if pred_label != true_label:
            print(f"  ❌ [MISSMATCH] True: {true_label} | Pred: {pred_label}")
            print(f"     Text: {text[:80]}...")

    # Calculate metrics using sklearn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "=" * 70)
    print("  Classification Metrics")
    print("=" * 70)
    print(f"  Accuracy  : {accuracy * 100:>6.2f} %")
    print(f"  Precision : {precision * 100:>6.2f} %")
    print(f"  Recall    : {recall * 100:>6.2f} %")
    print(f"  F1-Score  : {f1 * 100:>6.2f} %")
    print("=" * 70)
    
if __name__ == "__main__":
    # If a csv is passed via command line, use it
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "d:\\UG\\Call intrusion\\data\\test.csv"
    evaluate_model(csv_file)
