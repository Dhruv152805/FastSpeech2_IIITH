""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

Extended with Telugu phoneme inventory to support English and Telugu TTS.
"""

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness:
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Telugu phoneme set
_telugu_phones = [
    "a", "aa", "i", "ii", "u", "uu", "ri", "e", "ee", "ai", "o", "oo", "au",
    "k", "kh", "g", "gh", "ng", "ch", "chh", "j", "jh", "ny",
    "tt", "tth", "dd", "ddh", "nn", "t", "th", "d", "dh", "n",
    "p", "ph", "b", "bh", "m", "y", "r", "rr", "l", "ll", "lll",
    "v", "sh", "ss", "s", "h"
]
_geminates = [c + c for c in ["k", "kh", "g", "gh", "ch", "j", "tt", "dd", "t", "th", "d", "dh", "n", "nn", "p", "b", "m", "y", "r", "rr", "l", "ll", "v", "sh", "ss", "s", "h"]]
_telugu_all = sorted(set(_telugu_phones + _geminates))
_telugu = ["@" + s for s in _telugu_all]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _telugu
    + _silences
)
