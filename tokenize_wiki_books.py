import random
from collections import defaultdict
from multiprocessing import cpu_count

import nltk
from datasets import load_dataset, concatenate_datasets, set_caching_enabled
from transformers import GPT2TokenizerFast


def create_instances_from_document(tokenizer, document, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0

    segmented_sents = nltk.sent_tokenize(document)

    for i, sent in enumerate(segmented_sents):
        current_chunk.append(sent)
        current_length += len(tokenizer.tokenize(sent))
        if i == len(segmented_sents) - 1 or current_length >= max_seq_length:
            if len(current_chunk) > 1:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.append(current_chunk[j])

                tokens_b = []

                for j in range(a_end, len(current_chunk)):
                    tokens_b.append(current_chunk[j])

                if random.random() < 0.5:
                    # Random next
                    is_random_next = True
                    # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    # Actual next
                    is_random_next = False

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                instance = tokenizer(
                    ' '.join(tokens_a),
                    ' '.join(tokens_b),
                    truncation='longest_first',
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling
                    # is more efficient when it receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )
                assert len(instance['input_ids']) <= max_seq_length
                instance["sentence_order_label"] = 1 if is_random_next else 0
                instances.append(instance)

            current_chunk = []
            current_length = 0

    return instances


def tokenize_function(examples):
    # Remove empty texts
    texts = (text for text in examples["text"] if len(text) > 0 and not text.isspace())

    new_examples = defaultdict(list)

    for text in texts:
        instances = create_instances_from_document(tokenizer, text, max_seq_length=512)
        for instance in instances:
            for key, value in instance.items():
                new_examples[key].append(value)

    return new_examples


random.seed(0)
nltk.download('punkt')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

bookcorpus = load_dataset('bookcorpusopen', cache_dir='.data_cache')['train']
wikipedia = load_dataset('wikipedia', '20200501.en', cache_dir='.data_cache')['train']

set_caching_enabled(False)

tokenized_bookcorpus = bookcorpus.map(
    tokenize_function,
    batched=True,
    batch_size=4,
    num_proc=cpu_count(),
    remove_columns=["title", "text"],
)

tokenized_wiki = wikipedia.map(
    tokenize_function,
    batched=True,
    batch_size=4,
    num_proc=cpu_count(),
    remove_columns=["title", "text"],
)

wiki_bookcorpus = concatenate_datasets([bookcorpus, wikipedia], split='train')

wiki_bookcorpus.save_to_disk("gpt2_tokenized_data")