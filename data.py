import logging
from typing import Optional
from functools import partial
from datasets import interleave_datasets, IterableDataset

from lib.data.yt_streaming import YTDataset
from lib.data.preprocessing import tokenize_function, WrappedIterableDataset
logger = logging.getLogger(__name__)


def make_lazy_dataset(
    tokenizer,
    shuffle_buffer_size: int = 10 ** 4,
    shuffle_seed: Optional[int] = None,
    preprocessing_batch_size: int = 256,
    max_sequence_length: int = 512
):
    runet = YTDataset("hahn", "//home/yr/nlp/big_russian_bert/common_mincount1_nolimit_nodedup")
    wiki = YTDataset("hahn", "//home/yr/nlp/big_russian_bert/wikipedia")
    taiga = YTDataset("hahn", "//home/yr/nlp/big_russian_bert/taiga")
    librusec = YTDataset("hahn", "//home/yr/nlp/big_russian_bert/rdt")

    datasets = dict(runet=runet, wiki=wiki, taiga=taiga, librusec=librusec)
    weights = dict(runet=0.4, wiki=0.2, taiga=0.2, librusec=0.2)
    colnames = dict(runet=b"data", wiki=b"Text", taiga=b"text", librusec=b"text")

    def extract_training_columns(key, batch):
        texts = [bytes.decode(row, errors='ignore') for row in batch[colnames[key]]]
        return dict(text=texts, key=[key] * len(texts))

    datasets = {key: IterableDataset(dataset) for key, dataset in datasets.items()}
    datasets = {key: dataset.map(partial(extract_training_columns, key),
                                 batched=True, batch_size=preprocessing_batch_size)
                for key, dataset in datasets.items()}

    dataset = interleave_datasets([datasets[k] for k in sorted(datasets.keys())],
                                  probabilities=[weights[k] for k in sorted(datasets.keys())])

    dataset = dataset.map(partial(tokenize_function, tokenizer, max_sequence_length=max_sequence_length),
                          batched=True, batch_size=preprocessing_batch_size)

    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    dataset = dataset.with_format("torch")
    return WrappedIterableDataset(dataset)
