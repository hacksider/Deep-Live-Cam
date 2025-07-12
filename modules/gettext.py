import json
from pathlib import Path
from typing import Dict, Optional

class LanguageManager:
    """Manages language translations for the UI."""
    def __init__(self, default_language: str = "en"):
        self.current_language: str = default_language
        self.translations: Dict[str, str] = {}
        self.load_language(default_language)

    def load_language(self, language_code: str) -> bool:
        """Load a language file by code."""
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

    def _(self, key: str, default: Optional[str] = None) -> str:
        """Get translated text for a key."""
        return self.translations.get(key, default if default else key)