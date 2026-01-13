import re

def detect_language_strict(text, origin_dataset):
    """
    Strict Language Logic based on KRIXION Source.
    
    Args:
        text (str): The cleaned text sample.
        origin_dataset (str): Context ('english', 'hindi_mixed', 'indo_mixed').
        
    Returns:
        str: 'hi', 'en', or 'hi-en'
    """
    if not isinstance(text, str) or len(text) < 2: 
        return "en"
    
    # 1. Check for Devanagari (Hindi Script)
    # Unicode range for Devanagari: 0900-097F
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
    
    if devanagari_count > 2:
        return "hi"  # Text physically written in Hindi script
        
    # 2. If no Devanagari, check the Source Context
    # If text is Latin script but from a Code-Mixed source -> Hinglish
    if origin_dataset in ["hindi_mixed", "indo_mixed"]:
        return "hi-en" # Romanized Hindi (Hinglish)
        
    # 3. Default to English for everything else
    return "en" 