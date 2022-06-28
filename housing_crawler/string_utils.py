"""Functions and classes related to processing/manipulating strings"""
def remove_prefix(text, prefix):
    """Note that this method can just be replaced by str.removeprefix()
    if the project ever moves to Python 3.9+"""
    if text and text.startswith(prefix):
        return text[len(prefix):]
    return text
