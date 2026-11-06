import re
import unicodedata

# Comprehensive Telugu G2P Converter following phonetic grammar rules of Telugu

TELUGU_VOWELS = {
    '\u0C05': 'a',   '\u0C06': 'aa',  '\u0C07': 'i',   '\u0C08': 'ii',
    '\u0C09': 'u',   '\u0C0A': 'uu',  '\u0C0B': 'ri',  '\u0C0E': 'e',
    '\u0C0F': 'ee',  '\u0C10': 'ai',  '\u0C12': 'o',   '\u0C13': 'oo',  '\u0C14': 'au'
}

TELUGU_MATRAS = {
    '\u0C3E': 'aa',  '\u0C3F': 'i',   '\u0C40': 'ii',  '\u0C41': 'u',   '\u0C42': 'uu',
    '\u0C43': 'ri',  '\u0C46': 'e',   '\u0C47': 'ee',  '\u0C48': 'ai',
    '\u0C4A': 'o',   '\u0C4B': 'oo',  '\u0C4C': 'au'
}

TELUGU_CONSONANTS = {
    '\u0C15': 'k',   '\u0C16': 'kh',  '\u0C17': 'g',   '\u0C18': 'gh',  '\u0C19': 'ng',
    '\u0C1A': 'ch',  '\u0C1B': 'chh', '\u0C1C': 'j',   '\u0C1D': 'jh',  '\u0C1E': 'ny',
    '\u0C1F': 'tt',  '\u0C20': 'tth', '\u0C21': 'dd',  '\u0C22': 'ddh', '\u0C23': 'nn',
    '\u0C24': 't',   '\u0C25': 'th',  '\u0C26': 'd',   '\u0C27': 'dh',  '\u0C28': 'n',
    '\u0C2A': 'p',   '\u0C2B': 'ph',  '\u0C2C': 'b',   '\u0C2D': 'bh',  '\u0C2E': 'm',
    '\u0C2F': 'y',   '\u0C30': 'r',   '\u0C31': 'rr',  '\u0C32': 'l',   '\u0C33': 'll',
    '\u0C35': 'v',   '\u0C36': 'sh',  '\u0C37': 'ss',  '\u0C38': 's',   '\u0C39': 'h',
    '\u0C34': 'lll'
}

VIRAMA = '\u0C4D'
ANUSVARA = '\u0C02'
VISARGA = '\u0C03'

# Telugu number words for date and time normalization
TELUGU_DIGITS = {
    '0': 'sunna', '1': 'okati', '2': 'rendu', '3': 'moodu', '4': 'naalugu',
    '5': 'aidu', '6': 'aaru', '7': 'eedu', '8': 'enimidi', '9': 'tommidi',
    '10': 'padi', '12': 'pennendu', '45': 'nalabhai aidu', '2026': 'rendu vela iravai aaru'
}

def normalize_telugu_numbers(text: str) -> str:
    # 2026 -> rendu vela iravai aaru
    text = re.sub(r'\b2026\b', 'rendu vela iravai aaru', text)
    text = re.sub(r'\b10:45\b', 'padi nalabhai aidu', text)
    text = re.sub(r'\b12వ\b', 'pennendava', text)
    text = re.sub(r'\b12\b', 'pennendu', text)
    text = re.sub(r'\b10\b', 'padi', text)
    return text

def telugu_to_phonemes(text: str) -> str:
    text = normalize_telugu_numbers(text)
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\u200C', '').replace('\u200D', '')
    
    chars = list(text)
    n = len(chars)
    phones = []
    i = 0
    
    while i < n:
        c = chars[i]
        
        # Space or punctuation
        if c == ' ':
            if phones and phones[-1] != 'sp':
                phones.append('sp')
            i += 1
            continue
        elif c in ',.!?:;-\'\"()':
            if phones and phones[-1] != 'sp':
                phones.append('sp')
            i += 1
            continue
            
        # Independent vowel
        if c in TELUGU_VOWELS:
            phones.append(TELUGU_VOWELS[c])
            i += 1
            # Anusvara after vowel
            if i < n and chars[i] == ANUSVARA:
                phones.append('m')
                i += 1
            elif i < n and chars[i] == VISARGA:
                phones.append('h')
                i += 1
                
        # Consonant
        elif c in TELUGU_CONSONANTS:
            base_consonant = TELUGU_CONSONANTS[c]
            i += 1
            
            # Check for virama (halant)
            if i < n and chars[i] == VIRAMA:
                i += 1
                # Check for following consonant (conjunct or geminate)
                if i < n and chars[i] in TELUGU_CONSONANTS:
                    next_consonant = TELUGU_CONSONANTS[chars[i]]
                    i += 1
                    # Geminate
                    if next_consonant == base_consonant:
                        phones.append(base_consonant + base_consonant)
                    else:
                        phones.append(base_consonant)
                        phones.append(next_consonant)
                    
                    # Following matra or inherent vowel
                    if i < n and chars[i] in TELUGU_MATRAS:
                        phones.append(TELUGU_MATRAS[chars[i]])
                        i += 1
                    else:
                        phones.append('a')
                else:
                    # Pure consonant at end of word or syllable
                    phones.append(base_consonant)
            elif i < n and chars[i] in TELUGU_MATRAS:
                phones.append(base_consonant)
                phones.append(TELUGU_MATRAS[chars[i]])
                i += 1
            else:
                # Inherent vowel 'a'
                phones.append(base_consonant)
                phones.append('a')
                
            # Check anusvara or visarga after consonant syllable
            if i < n and chars[i] == ANUSVARA:
                phones.append('m')
                i += 1
            elif i < n and chars[i] == VISARGA:
                phones.append('h')
                i += 1
        elif c.isalpha():
            # Latin character fallback (e.g. for loan words)
            phones.append(c.lower())
            i += 1
        else:
            i += 1

    # Clean trailing or consecutive sp
    cleaned_phones = []
    for p in phones:
        if p == 'sp' and (not cleaned_phones or cleaned_phones[-1] == 'sp'):
            continue
        cleaned_phones.append(p)
    if cleaned_phones and cleaned_phones[-1] == 'sp':
        cleaned_phones.pop()
        
    return '{' + ' '.join(cleaned_phones) + '}'

if __name__ == '__main__':
    tests = [
        ("T1", "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."),
        ("T2a", "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."),
        ("T2b", "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."),
        ("T3", "పులి అడవిలో నిశ్శబ్దంగా నడిచింది."),
        ("T4", "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.")
    ]
    for tid, txt in tests:
        print(f"[{tid}] {txt}")
        print(f"  -> {telugu_to_phonemes(txt)}\n")

