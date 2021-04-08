import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import transformers
from apex.optimizers import FusedLAMB
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, Trainer, TrainingArguments, set_seed, \
    AlbertTokenizerFast, AlbertConfig, AlbertForPreTraining
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process

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
    dataset_path: Optional[str] = field(
        default='.', metadata={"help": "Path to the dataset"}
    )

    tokenizer_path: Optional[str] = field(
        default='.', metadata={"help": "Path to the dataset"}
    )

    config_path: Optional[str] = field(
        default='.', metadata={"help": "Path to the dataset"}
    )
    cache_dir: Optional[str] = field(
        default='.', metadata={"help": "Path to the cache"}
    )


def main():
    parser = HfArgumentParser((TrainingArguments, DatasetArguments))
    training_args, dataset_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AlbertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)

    tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)

    # find latest checkpoint in output_dir
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint_dir = max(output_dir.glob('checkpoint*'), default=None, key=os.path.getctime)

    if latest_checkpoint_dir is not None:
        logger.info(f'Loading model from {latest_checkpoint_dir}')
        model = AlbertForPreTraining.from_pretrained(latest_checkpoint_dir)
    else:
        logger.info(f'Training from scratch')
        model = AlbertForPreTraining(config)
        model.resize_token_embeddings(len(tokenizer))

    tokenized_dataset_path = Path(dataset_args.dataset_path)

    tokenized_datasets = load_from_disk(tokenized_dataset_path)

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

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

    optimizer = FusedLAMB(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    trainer = NirvanaCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler)
    )

    # Training
    if training_args.do_train:
        trainer.train(model_path=latest_checkpoint_dir)


if __name__ == "__main__":
    main()
