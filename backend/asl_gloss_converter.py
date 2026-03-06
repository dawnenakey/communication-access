"""
SonZo AI - English to ASL Gloss Converter
Patent-pending Sign Language Recognition technology
Dawnena Key - dawnena@sonzo.io
"""

# Comprehensive English -> ASL gloss mapping
WORD_MAP = {
    # Skip words (no sign equivalent)
    "the": None, "a": None, "an": None, "is": None, "are": None,
    "am": None, "was": None, "were": None, "be": None, "been": None,
    "of": None, "to": None, "in": None, "it": None, "its": None,
    "that": "THAT", "this": "THIS", "with": None, "for": "FOR",
    "on": None, "at": None, "by": None, "from": None, "as": None,

    # Pronouns
    "i": "ME", "i/me": "ME", "me": "ME", "my": "MY", "mine": "MY",
    "you": "YOU", "your": "YOUR", "yours": "YOUR",
    "we": "WE", "our": "WE", "us": "WE",
    "they": None, "them": None, "their": None,
    "he": None, "she": None, "him": None, "her": None,

    # Greetings
    "hello": "HELLO", "hi": "HELLO", "hey": "HELLO",
    "goodbye": "GOODBYE", "bye": "GOODBYE", "later": "GOODBYE",
    "welcome": "WELCOME",
    "nice": "NICE_TO_MEET_YOU", "meet": "NICE_TO_MEET_YOU",
    "pleased": "PLEASURE", "pleasure": "PLEASURE",

    # Common verbs
    "help": "HELP", "helping": "HELP", "helps": "HELP",
    "understand": "UNDERSTAND", "understanding": "UNDERSTAND", "understood": "UNDERSTAND",
    "learn": "LEARN", "learning": "LEARN", "learned": "LEARN",
    "know": "KNOW", "knowing": "KNOW", "knew": "KNOW",
    "think": "THINK", "thinking": "THINK",
    "like": "LIKE", "likes": "LIKE", "liked": "LIKE",
    "love": "LOVE", "loves": "LOVE", "loved": "LOVE",
    "want": "WANT", "wants": "WANT", "wanted": "WANT",
    "need": "NEED", "needs": "NEED", "needed": "NEED",
    "go": "GO", "going": "GO", "went": "GO",
    "come": "COME", "coming": "COME", "came": "COME",
    "stop": "STOP", "stopping": "STOP", "stopped": "STOP",
    "wait": "WAIT", "waiting": "WAIT",
    "work": "WORK", "working": "WORK", "worked": "WORK",
    "eat": "EAT", "eating": "EAT", "ate": "EAT",
    "drink": "DRINK", "drinking": "DRINK", "drank": "DRINK",
    "sleep": "SLEEP", "sleeping": "SLEEP", "slept": "SLEEP",
    "communicate": "COMMUNICATE", "communication": "COMMUNICATE",
    "recognize": "RECOGNIZES", "recognizes": "RECOGNIZES", "recognition": "RECOGNITION",
    "assist": "ASSIST", "assisting": "ASSIST",
    "answer": "ANSWER", "answers": "ANSWER",
    "finish": "FINISH", "finished": "FINISH", "done": "FINISH",
    "again": "AGAIN", "repeat": "AGAIN",

    # Adjectives
    "good": "GOOD", "great": "GREAT", "nice": "NICE_TO_MEET_YOU",
    "bad": "BAD", "happy": "HAPPY", "sad": "SAD",
    "accurate": "ACCURATE", "full": "FULL",
    "today": "TODAY",

    # Questions
    "what": "WHAT", "where": "WHERE", "who": "WHO",
    "why": "WHY", "how": "HOW", "when": "WHEN",

    # Yes/No
    "yes": "YES", "no": "NO", "not": "NO",
    "please": "PLEASE", "sorry": "SORRY", "thank": "THANK_YOU",
    "thanks": "THANK_YOU", "okay": "YES", "ok": "YES",

    # Tech/App specific
    "app": "APP", "application": "APP",
    "sign": "SIGN", "signing": "SIGN", "signed": "SIGN",
    "language": "LANGUAGE", "languages": "LANGUAGE",
    "ai": "AI", "artificial": "AI",
    "data": "DATA", "depth": "DEPTH",
    "camera": "CAMERA", "captures": "CAPTURES",
    "sentences": "SENTENCES", "sentence": "SENTENCES",
    "using": "USING", "use": "USING",
    "more": "MORE", "name": "NAME",
    "can": "CAN", "here": "HERE",
}

def english_to_asl_gloss(text: str) -> list:
    """
    Convert English text to ASL gloss word list.
    Returns list of uppercase gloss words.
    """
    # Clean and split
    words = text.lower()
    for char in ".,!?;:\"'()-":
        words = words.replace(char, " ")
    words = words.split()

    gloss = []
    for word in words:
        if word in WORD_MAP:
            mapped = WORD_MAP[word]
            if mapped:  # None means skip
                gloss.append(mapped)
        else:
            # Unknown word - use uppercase as-is
            gloss.append(word.upper())

    return gloss

if __name__ == "__main__":
    tests = [
        "hello how are you",
        "this app recognizes sign language",
        "I need help understanding",
        "thank you for learning sign language with me",
    ]
    for t in tests:
        print(f"Input:  {t}")
        print(f"Gloss:  {' '.join(english_to_asl_gloss(t))}")
        print()
