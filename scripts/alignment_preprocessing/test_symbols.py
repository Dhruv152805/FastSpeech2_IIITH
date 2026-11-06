import os
import sys

# Telugu phoneme tokens derived from aksharas and consonants
telugu_phones = [
    'a', 'aa', 'i', 'ii', 'u', 'uu', 'ri', 'e', 'ee', 'ai', 'o', 'oo', 'au',
    'k', 'kh', 'g', 'gh', 'ng', 'ch', 'chh', 'j', 'jh', 'ny',
    'tt', 'tth', 'dd', 'ddh', 'nn', 't', 'th', 'd', 'dh', 'n',
    'p', 'ph', 'b', 'bh', 'm', 'y', 'r', 'rr', 'l', 'll', 'lll',
    'v', 'sh', 'ss', 's', 'h'
]
geminates = [c + c for c in ['k', 'g', 'ch', 'j', 'tt', 'dd', 't', 'd', 'n', 'p', 'b', 'm', 'y', 'r', 'l', 'v', 's']]
telugu_all = sorted(set(telugu_phones + geminates))

sys.path.insert(0, r"C:\Users\dkhan\Desktop\fastspeech2-iiith\reference_fastspeech2")
from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]
_telugu = ["@" + s for s in telugu_all]

all_symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _telugu
    + _silences
)

print(f"Total symbols: {len(all_symbols)}")
print(f"Telugu symbols count: {len(_telugu)}")
print("Sample Telugu symbols:", _telugu[:10])

