import logging
import math
import os
import sys
from dataclasses import dataclass, field
from multiprocessing import cpu_count

import transformers
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, Trainer, TrainingArguments, set_seed, \
    RobertaTokenizerFast
from transformers.trainer_utils import is_main_process
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

from configuration_roberta_fixup import RobertaFixupConfig
from modeling_roberta_fixup import RobertaForMaskedLM

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    use_fixup: bool = field(default=False)

    ln_type: str = field(default='post')


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

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

    config = RobertaFixupConfig.from_pretrained('roberta-large', use_fixup=model_args.use_fixup,
                                                ln_type=model_args.ln_type)

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    model = RobertaForMaskedLM(config)

    model.resize_token_embeddings(len(tokenizer))

    if not os.path.exists('tokenized_wikitext'):
        wikitext = load_dataset('wikitext', 'wikitext-103-v1', cache_dir='.data_cache')

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                max_length=128,
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

        tokenized_datasets.save_to_disk('tokenized_wikitext')
    else:
        tokenized_datasets = load_from_disk('tokenized_wikitext')

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

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    # if training_args.warmup_steps != 0:
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )
    # else:
    #     lr_scheduler = get_constant_schedule(
    #         optimizer
    #     )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler)
    )

    # Training
    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    main()
