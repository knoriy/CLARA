from transformers import GPT2TokenizerFast
import os

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "zh-CN":"chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "iw": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "gn": "Guarani",
    "fy": "western frisian",
    "fy-NL":"western frisian (netherlands)",
    "eo": "Esperanto",
    "cnh": "Hakha Chin",
    "cv": "Chuvash",
    "dv": "Divehi",
    "ky": "Kirghiz",
    "or": "Oriya",
    "rw": "Kinyarwanda",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "british": "en",
    "american": "en",
    "Dhivehi": "dv",
    "Maldivian": "dv",
    "farsi": "fa",
    "Kyrgyz": "ky",
    "Odia": "or",
    "Slovene": "sl",


}



class Tokeniser(GPT2TokenizerFast):
    def __init__(self) -> None:
        self.tokeniser = self.get_tokeniser()

    def encode(self, text, language:str="en", speech:bool=True, **kwargs):
        text = self.preprocess_text(text, language, speech)
        return self.tokeniser.encode(text)

    def decode(self, tokens, **kwargs):
        return self.tokeniser.decode(tokens)

    def preprocess_text(self, text, language:str="en", speech:bool=True):
        if language:
            language = language.lower()
            if language not in LANGUAGES:
                if language in TO_LANGUAGE_CODE:
                    language = TO_LANGUAGE_CODE[language]
                else:
                    raise ValueError(f"Unsupported language: {language} \
                    \nSuported languages are: \n{list(LANGUAGES.values())}")

        specials = [
            f"<|{language}|>",
            "<|speech|>" if speech else "<|nospeech|>"

        ]
        text = "".join([*specials, " ", text])
        return text
    
    def get_vocab_size(self) -> int:
        return len(self.tokeniser.get_vocab())

    def get_tokeniser(name:str = "multilingual"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        path = os.path.join(os.path.dirname(__file__), "whisper/assets", name)
        tokenizer = GPT2TokenizerFast.from_pretrained(path)

        specials = [
            "<|startoftranscript|>",
            *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
            "<|translate|>",
            "<|transcribe|>",
            # "<|startoflm|>",
            # "<|startofprev|>",
            "<|nospeech|>",
            "<|speech|>",
            # "<|notimestamps|>",
        ]

        tokenizer.add_special_tokens(dict(additional_special_tokens=specials))
        return tokenizer