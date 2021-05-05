import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk
from hivemind import DHT
from torch_optimizer import Lamb
from transformers import get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, HfArgumentParser, \
    Trainer, TrainingArguments, AlbertTokenizerFast, AlbertConfig
from transformers.optimization import get_linear_schedule_with_warmup

from modeling_albert_moe import AlbertForPreTraining

logger = logging.getLogger(__name__)


class NirvanaCheckpointTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        if self.is_world_process_zero():
            try:
                import nirvana_dl.snapshot as snap
                snap.dump_snapshot()
                logger.info('Checkpoint saved to snapshots.')
            except Exception as e:
                logger.info(f'Checkpoint not saved to snapshots: {e}')


@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    tokenizer_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    config_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    cache_dir: Optional[str] = field(default='.', metadata={"help": "Path to the cache"})


def main(dataset_args, args):
    tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)

    tokenized_datasets = load_from_disk(dataset_args.dataset_path)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=not args.lm)

    grid_size = (args.grid_size,)
    dht = DHT(initial_peers=[args.init_peer], start=True, listen=False, max_workers=4, parallel_rpc=16,
              wait_timeout=0.1)

    config = AlbertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)

    model = AlbertForPreTraining(config, grid_size, dht)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = Lamb(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=1e-6,
        weight_decay=training_args.weight_decay,
        clamp_value=10000,
        debias=True,
    )
    # TODO DecentralizedOptimizer

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    # TODO correct save/load
    if args.restore_from_checkpoint is not None:
        checkpoint_path = Path(args.restore_from_checkpoint) / 'pytorch_model.bin'
        if checkpoint_path.exists():
            model_checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(model_checkpoint)

    trainer = NirvanaCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(optimizer, lr_scheduler)
    )

    trainer.train(args.restore_from_checkpoint)


if __name__ == '__main__':
    parser = HfArgumentParser((DatasetArguments, TrainingArguments))
    parser.add_argument('--grid-size', type=int, default=8)
    parser.add_argument('--lm', action='store_true')
    parser.add_argument('--init-peer')

    dataset_args, training_args, args = parser.parse_args_into_dataclasses()
    main(dataset_args, training_args, args)
