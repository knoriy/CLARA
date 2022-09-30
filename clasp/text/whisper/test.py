from normalizers import EnglishTextNormalizer
from tokenizer import get_tokenizer

from transformers import GPT2TokenizerFast, GPT2Model
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# if __name__ == '__main__':
#     cleaner = EnglishTextNormalizer()
#     cleaned_text = cleaner("Hello, how are you doing? this is Normalised test! {some other text}, won't, been, re, $100 - $100")
#     tokenizer = get_tokenizer(True)
