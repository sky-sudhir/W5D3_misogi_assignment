import wikipedia

def fetch_from_wikipedia(topic: str, lang="en", max_sentences=5) -> str:
    try:
        wikipedia.set_lang(lang)
        summary = wikipedia.summary(topic, sentences=max_sentences)
        return f"🧠 Wikipedia Summary for '{topic}':\n{summary}"
    except Exception as e:
        return f"(Wikipedia lookup failed: {str(e)})"
