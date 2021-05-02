from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_from_disk
from hivemind import RemoteSwitchMixtureOfExperts, DHT
from torch_optimizer import Lamb
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, HfArgumentParser, \
    Trainer, TrainingArguments, AlbertTokenizerFast, AlbertConfig
from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertMLMHead, MaskedLMOutput
from transformers.optimization import get_linear_schedule_with_warmup


@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})

    tokenizer_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})

    config_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    cache_dir: Optional[str] = field(default='.', metadata={"help": "Path to the cache"})


class TrainerModel(nn.Module):
    def __init__(self, vocab_size, expert_dim, grid_size, dht, num_moe_blocks=1):
        super().__init__()
        self.config = AlbertConfig.from_pretrained('albert-base-v2')
        self.config.hidden_size = expert_dim
        self.config.vocab_size = vocab_size
        self.embedding_hidden_mapping_in = nn.Linear(self.config.embedding_size, self.config.hidden_size)

        self.embeddings = AlbertEmbeddings(self.config)

        self.mixture = nn.ModuleList(
            [
                RemoteSwitchMixtureOfExperts(in_features=expert_dim, grid_size=grid_size, dht=dht,
                                             k_best=1, k_min=0, forward_timeout=2, backward_timeout=2,
                                             timeout_after_k_min=1, detect_anomalies=True,
                                             uid_prefix='expert.')
                for _ in range(num_moe_blocks)])

        self.lm_head = AlbertMLMHead(self.config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        batch_size, seq_len = input_ids.size()
        embeddings = self.embedding_hidden_mapping_in(self.embeddings(input_ids))

        loss_per_mixture = []

        for layer in self.mixture:
            # reshape and reshape back for token-level experts
            embeddings = embeddings.view(batch_size * seq_len, -1)
            embeddings, balancing_loss = layer(embeddings)
            loss_per_mixture.append(balancing_loss)
            embeddings = embeddings.view(batch_size, seq_len, -1)

        prediction_scores = self.lm_head(embeddings)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = nn.functional.cross_entropy(prediction_scores.view(-1, self.config.vocab_size),
                                                         labels.view(-1))
            total_loss = masked_lm_loss + torch.stack(loss_per_mixture).sum()

        if not return_dict:
            output = (prediction_scores, embeddings)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=total_loss,
            logits=prediction_scores,
        )


def main(dataset_args, args):
    tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)

    tokenized_datasets = load_from_disk(dataset_args.dataset_path)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=not args.lm)

    training_args = TrainingArguments(output_dir='lm', overwrite_output_dir=True,
                                      do_train=True, do_eval=True, do_predict=False,
                                      prediction_loss_only=True,
                                      per_device_train_batch_size=4,
                                      per_device_eval_batch_size=4, max_steps=10000, warmup_steps=4000,
                                      gradient_accumulation_steps=1,
                                      logging_first_step=True, logging_steps=10, max_grad_norm=1,
                                      weight_decay=0.01,
                                      save_total_limit=5, seed=0, dataloader_num_workers=0, no_cuda=not args.cuda)

    grid_size = (args.grid_size,)
    dht = DHT(initial_peers=[args.init_peer], start=True, listen=False, max_workers=4, parallel_rpc=16,
              wait_timeout=0.1)

    model = TrainerModel(tokenizer.vocab_size, args.hidden_dim, grid_size, dht)

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

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    if args.restore_from_checkpoint is not None:
        checkpoint_path = Path(args.restore_from_checkpoint) / 'pytorch_model.bin'
        if checkpoint_path.exists():
            model_checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(model_checkpoint)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets,
                      tokenizer=tokenizer,
                      data_collator=collator,
                      optimizers=(optimizer, lr_scheduler))

    print('Starting training')
    trainer.train(args.restore_from_checkpoint)


if __name__ == '__main__':
    parser = HfArgumentParser((DatasetArguments,))
    parser.add_argument('--hidden-dim', type=int, default=1024, help='main dimension for expert_cls')
    parser.add_argument('--grid-size', type=int, default=8)
    parser.add_argument('--lm', action='store_true')
    parser.add_argument('--init-peer')
    parser.add_argument('--restore-from-checkpoint')
    parser.add_argument('--cuda', action='store_true')
    dataset_args, args = parser.parse_args_into_dataclasses()
    main(dataset_args, args)
