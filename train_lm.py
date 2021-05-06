import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
from packaging import version
import torch

from datasets import load_from_disk
from hivemind import DHT
from torch_optimizer import Lamb
from transformers import get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, HfArgumentParser, \
    TrainingArguments, AlbertTokenizerFast, AlbertConfig
from transformers.trainer import Trainer
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

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        mlm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        balancing_loss = outputs["balancing_loss"] if isinstance(outputs, dict) else outputs[1]
        return mlm_loss + self.args.balancing_loss_weight * balancing_loss


@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    tokenizer_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    config_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    cache_dir: Optional[str] = field(default='.', metadata={"help": "Path to the cache"})


@dataclass
class MixtureTrainingArguments(TrainingArguments):
    balancing_loss_weight: Optional[float] = field(default=1e-2)


def main(dataset_args, training_args, args):
    tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)

    tokenized_datasets = load_from_disk(dataset_args.dataset_path)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=not args.lm)

    grid_size = (args.grid_size,)
    dht = DHT(initial_peers=[args.init_peer], start=True, listen=False, max_workers=16, parallel_rpc=32,
              wait_timeout=2)

    config = AlbertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)

    # find latest checkpoint in output_dir
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint_dir = max(output_dir.glob('checkpoint*'), default=None, key=os.path.getctime)

    if latest_checkpoint_dir is not None:
        logger.info(f'Loading model from {latest_checkpoint_dir}')
        model = AlbertForPreTraining.from_pretrained(latest_checkpoint_dir, grid_size=grid_size, dht=dht)
    else:
        logger.info(f'Training from scratch')
        model = AlbertForPreTraining(config, grid_size, dht)
        model.resize_token_embeddings(len(tokenizer))

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

    trainer = NirvanaCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(optimizer, lr_scheduler)
    )

    trainer.train(resume_from_checkpoint=latest_checkpoint_dir)


if __name__ == '__main__':
    parser = HfArgumentParser((DatasetArguments, MixtureTrainingArguments))
    parser.add_argument('--grid-size', type=int, required=True)
    parser.add_argument('--lm', action='store_true')
    parser.add_argument('--init-peer', required=True)

    dataset_args, training_args, args = parser.parse_args_into_dataclasses()
    main(dataset_args, training_args, args)
