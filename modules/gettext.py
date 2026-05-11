import json
from pathlib import Path

RTL_LANGUAGES = {"ar", "fa", "he", "ur"}
RTL_MARK = "\u200f"


class LanguageManager:
    def __init__(self, default_language="en"):
        self.current_language = default_language
        self.translations = {}
        self.load_language(default_language)

    def load_language(self, language_code) -> bool:
        """load language file"""
        if language_code == "en":
            return True
        try:
            file_path = Path(__file__).parent.parent / f"locales/{language_code}.json"
            with open(file_path, "r", encoding="utf-8") as file:
                self.translations = json.load(file)
            self.current_language = language_code
            return True
        except FileNotFoundError:
            print(f"Language file not found: {language_code}")
            return False

    def _(self, key, default=None) -> str:
        """Get translated text"""
        text = self.translations.get(key, default if default is not None else key)
        normalized_lang = str(self.current_language).lower().replace("_", "-").split("-", 1)[0]
        if normalized_lang in RTL_LANGUAGES:
            stripped_text = text.lstrip()
            if stripped_text.startswith(RTL_MARK):
                return text
            leading_whitespace = text[: len(text) - len(stripped_text)]
            return f"{leading_whitespace}{RTL_MARK}{stripped_text}"
        return text
