from multiprocessing import cpu_count

from datasets import load_dataset
from transformers import AlbertTokenizerFast

tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')

wikitext = load_dataset('wikitext', 'wikitext-103-v1', cache_dir='.data_cache')


def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=512,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        return_special_tokens_mask=True,
    )


tokenized_datasets = wikitext.map(
    tokenize_function,
    batched=True,
    num_proc=cpu_count(),
    remove_columns=["text"],
)

tokenized_datasets.save_to_disk('albert_tokenized_wikitext')
