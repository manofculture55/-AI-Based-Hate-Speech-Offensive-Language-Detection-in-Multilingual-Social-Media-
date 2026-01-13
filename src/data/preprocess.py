import re

def clean_text(text):
    """
    Mandatory Preprocessing Pipeline [Section 4]:
    - Remove URLs
    - Remove User Mentions (@user)
    - Remove Hashtags (#topic)
    - Remove extra whitespace
    """
    if not isinstance(text, str): 
        return ""
    
    # Remove URLs (http/https/www)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @mentions and #hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Remove extra spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
