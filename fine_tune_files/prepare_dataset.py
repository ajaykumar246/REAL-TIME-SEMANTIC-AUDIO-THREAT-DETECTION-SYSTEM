"""
Dataset Preparation for Multilingual Spam Classification.

Downloads the UCI SMS Spam Collection and augments it with
synthetic Tamil / Tanglish samples, then saves train/test splits.

Usage:
    python prepare_dataset.py
"""

import os
import csv
import urllib.request
import zipfile
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, FINETUNE_TEST_SIZE

# ──────────────────────────────────── constants
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00228/smsspamcollection.zip"
)
RAW_ZIP = os.path.join(DATA_DIR, "smsspamcollection.zip")
RAW_TSV = os.path.join(DATA_DIR, "SMSSpamCollection")

# ──────────────────────────────────── synthetic data
# Tamil / Tanglish spam samples (common patterns in Indian spam calls)
SYNTHETIC_SPAM_TAMIL = [
    "வாடிக்கையாளரே, உங்கள் கணக்கில் ₹50,000 பரிசு உள்ளது. உடனடியாக அழைக்கவும்.",
    "Sir, ungalukku oru special offer irukku. Bank loan 2% interest la kedaikum.",
    "Congratulations! Neenga oru lucky winner. Prize claim panna 9876543210 call pannunga.",
    "உங்கள் credit card-la cashback offer irukku. Innaiku last date sir.",
    "Madam, insurance scheme romba nalla irukku. Monthly 500 Rs mattum.",
    "வணக்கம், உங்கள் bank account verify pannanum. OTP sollunga please.",
    "Sir, personal loan thevaiyna? 1 lakh to 10 lakh varai instant approval.",
    "Ungalukku free mobile recharge kedaikum. Ipo click pannunga.",
    "தங்கள் கணக்கில் suspicious activity found. Udane bank-ku call pannunga.",
    "Hello sir, investment opportunity irukku. Monthly 20% return guaranteed.",
    "Neenga selected aairukeenga! Car prize win panna link-la click pannunga.",
    "Sir congratulations, bank account la ₹1,00,000 credit aagum 2 days la.",
    "Madam romba nalla scheme irukku, limited time offer, miss pannatheenga.",
    "உங்கள் phone number lucky draw-la select aachu. Prize: Gold chain!",
    "Urgent: Account block aagum, verify panna ipo call pannunga sir.",
    "Vanakkam, credit card upgrade offer. Annual fee free for lifetime.",
    "Sir, home loan lowest interest rate. Apply panna missed call pannunga.",
    "Ungal mobile-ku free data pack. Ipo activate pannunga - limited offer.",
    "Special diwali offer: All loans at 0% processing fee, call now sir.",
    "Vangalam sir, mutual fund la invest pannunga, guaranteed returns.",
    "Kedaikum madam, 50% discount on insurance premium, don't miss.",
    "Sir neenga winner! Bike prize claim panna idha click pannunga.",
    "Panam thevaiyna? Instant cash loan 30 minutes la account la varum.",
    "Ungalukku selected aagirukkeenga! Free trip to Goa, call pannunga.",
    "Madam, gold rate today lowest. Nalla chance, invest pannunga.",
    "Bank officer speaking sir, ungal account security check pannanum.",
    "Congratulations madam! Shopping voucher worth ₹10,000 ungalukku.",
    "Sir kaasu venuma? Personal loan no documents required, apply now.",
    "Lucky customer neenga! iPhone 15 win panna register pannunga.",
    "Ungal electricity bill-la overcharge irukku. Refund panna call pannunga.",
]

SYNTHETIC_HAM_TAMIL = [
    "நாளை எப்போ சந்திக்கலாம்? காலை 10 மணிக்கு ஓகேவா?",
    "Naan late varuvena, 15 mins wait pannunga.",
    "Amma kitta phone pannuda, urgent-a pesanum.",
    "Tomorrow class cancel-nu teacher sonna, FYI.",
    "Rice vangitu va varapo, 5 kg.",
    "Bro, cricket match paakka pogalama? Night plan pannuvom.",
    "என்ன appa saptu mudicha? Naan innum office la.",
    "Assignment submit panna last date yapo? Reminder kudu.",
    "Exam results site la vandhurukku, check pannuda.",
    "Ipo veetla irukiya? Naan 5 minutes la varuven.",
    "ஹாய், என்ன நலமா? வீட்ல எல்லாரும் நலமா?",
    "Meeting 3 PM ku postpone aachu, note pannu.",
    "Bus late, traffic romba worst, sorry da.",
    "Birthday party Sunday evening 6 PM, varuya?",
    "Medicine edutha? Doctor 2 times nu sonnaru.",
    "Naan station la irukken, innum 10 mins la varuven.",
    "Coffee kudikka pogalama? Break time la.",
    "Grocery list: milk, bread, eggs, tomato.",
    "Movie ticket book pannirukkena, evening 7 show.",
    "Rent pay panniten, receipt screenshot anupuren.",
    "Temple poganum weekend la, plan pannuvom.",
    "Photo nalla vandhurukku da, super click.",
    "Wi-Fi password enna? Maranthuduchen.",
    "Libraryila book return panna maranthuduven, remind pannu.",
    "Match score enna? Naan paakka mudiyala.",
    "Train ticket confirm aachu, PNR forward pannuren.",
    "Project presentation ready aa? Last minute changes irukka?",
    "Paati phone number kudu, wish pannanum birthday.",
    "Laptop charger konduva, ennoda dead aachu.",
    "Dinner outside pogalama? Pizza hut or biryani?",
]

SYNTHETIC_SPAM_ENGLISH = [
    "Dear customer, your account has been selected for a cash prize of $5000. Call now!",
    "URGENT: Your bank account will be suspended. Verify immediately at this link.",
    "Congratulations! You've won a brand new iPhone 15. Claim your prize today!",
    "Limited time offer: Get a personal loan at 0% interest. Apply now!",
    "Your credit score qualifies you for an exclusive platinum card. No annual fee!",
    "Act now! Free insurance coverage worth $100,000. Limited slots available.",
    "Sir/Madam, this is calling from the bank regarding your pending loan approval.",
    "You have been specially selected for our premium investment scheme. Guaranteed 25% returns!",
    "ALERT: Suspicious activity detected on your account. Call us immediately to secure it.",
    "Win a luxury vacation to Maldives! Just answer 3 simple questions.",
    "Dear valued customer, your loan has been pre-approved. No documents needed.",
    "Final notice: Claim your tax refund of $2,500 before it expires.",
    "Exclusive offer for you: Buy one get one free on all credit cards!",
    "Your phone number has won our weekly lucky draw. Prize: $10,000 cash!",
    "Important: Your insurance policy needs immediate renewal. Call this number.",
]

SYNTHETIC_HAM_ENGLISH = [
    "Hey, are you free for lunch tomorrow? Let's catch up.",
    "Can you pick up some groceries on your way home?",
    "Meeting rescheduled to 4 PM. Please update your calendar.",
    "Happy birthday! Wishing you the best year ahead.",
    "Don't forget to submit the report by Friday.",
    "Running late, traffic is terrible. Be there in 20 mins.",
    "Did you check the exam results? They're out on the website.",
    "Let me know when you reach home safely.",
    "Can you send me the notes from today's lecture?",
    "Movie tonight? I'll book the tickets if you're in.",
    "Reminder: Doctor's appointment at 3 PM tomorrow.",
    "Thanks for dinner last night, it was great!",
    "The package arrived, I'll open it when you get here.",
    "What time does the store close? I need to grab something.",
    "Good morning! Have a productive day at work.",
]


def download_uci_dataset() -> pd.DataFrame:
    """Download and parse the UCI SMS Spam Collection."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(RAW_TSV):
        print("[*] Downloading UCI SMS Spam Collection ...")
        urllib.request.urlretrieve(UCI_URL, RAW_ZIP)
        with zipfile.ZipFile(RAW_ZIP, "r") as zf:
            zf.extractall(DATA_DIR)
        print("[+] Download complete.")

    rows = []
    with open(RAW_TSV, encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                label = 1 if row[0].strip().lower() == "spam" else 0
                rows.append({"text": row[1].strip(), "label": label})

    df = pd.DataFrame(rows)
    print(f"[+] UCI dataset loaded: {len(df)} samples "
          f"(spam={df['label'].sum()}, ham={len(df)-df['label'].sum()})")
    return df


def build_synthetic_samples() -> pd.DataFrame:
    """Create synthetic Tamil / Tanglish + extra English samples."""
    rows = []
    for text in SYNTHETIC_SPAM_TAMIL + SYNTHETIC_SPAM_ENGLISH:
        rows.append({"text": text, "label": 1})
    for text in SYNTHETIC_HAM_TAMIL + SYNTHETIC_HAM_ENGLISH:
        rows.append({"text": text, "label": 0})

    df = pd.DataFrame(rows)
    print(f"[+] Synthetic dataset: {len(df)} samples "
          f"(spam={df['label'].sum()}, ham={len(df)-df['label'].sum()})")
    return df


def main():
    """Download, augment, split, and save the full dataset."""
    uci_df = download_uci_dataset()
    synth_df = build_synthetic_samples()

    # Combine
    combined = pd.concat([uci_df, synth_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[+] Combined dataset: {len(combined)} samples "
          f"(spam={combined['label'].sum()}, "
          f"ham={len(combined)-combined['label'].sum()})")

    # Split
    train_df, test_df = train_test_split(
        combined,
        test_size=FINETUNE_TEST_SIZE,
        random_state=42,
        stratify=combined["label"],
    )

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[+] Saved: {train_path} ({len(train_df)} rows)")
    print(f"[+] Saved: {test_path}  ({len(test_df)} rows)")
    print("[✓] Dataset preparation complete!")


if __name__ == "__main__":
    main()
