import sys, os
sys.path.insert(0, r"C:\Users\dkhan\Desktop\fastspeech2-iiith\reference_fastspeech2")
from text.cleaners import english_cleaners

sentences = [
    ("E1", "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."),
    ("E2a", "The orchestra's conduct during the ceremony was praised by the audience."),
    ("E2b", "The researchers will conduct an experiment to evaluate the proposed method."),
    ("E3", "The rhyth m of the orchestra changed unexpectedly."),
    ("E4", "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.")
]

for name, s in sentences:
    cleaned = english_cleaners(s)
    print(f"{name} Original: {s}")
    print(f"{name} Cleaned : {cleaned}\n")

