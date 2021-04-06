import os
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import datasets
import torch
import torch.nn as nn
from hivemind import RemoteMixtureOfExperts, DHT
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.albert import AlbertTokenizerFast
from transformers.models.albert.modeling_albert import AlbertEmbeddings, AlbertMLMHead, AlbertConfig, MaskedLMOutput


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
                RemoteMixtureOfExperts(in_features=expert_dim, grid_size=grid_size, dht=dht,
                                       k_best=3, k_min=2, forward_timeout=5, backward_timeout=5,
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

        for layer in self.mixture:
            # reshape and reshape back for token-level experts
            embeddings = embeddings.view(batch_size * seq_len, -1)
            embeddings = layer(embeddings)
            embeddings = embeddings.view(batch_size, seq_len, -1)
        prediction_scores = self.lm_head(embeddings)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = nn.functional.cross_entropy(prediction_scores.view(-1, self.config.vocab_size),
                                                         labels.view(-1))

        if not return_dict:
            output = (prediction_scores, embeddings)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
        )


def main(args):
    torch.set_num_threads(16)
    tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')

    if not os.path.exists('tokenized_wikitext'):
        wikitext = datasets.load_dataset('wikitext', 'wikitext-103-v1', cache_dir='.data_cache')

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                max_length=256,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_wikitext = wikitext.map(
            tokenize_function,
            batched=True,
            num_proc=cpu_count(),
            remove_columns=["text"],
        )

        tokenized_wikitext.save_to_disk('tokenized_wikitext')
    else:
        tokenized_wikitext = datasets.load_from_disk('tokenized_wikitext')

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(output_dir='lm', overwrite_output_dir=True,
                                      do_train=True, do_eval=True, do_predict=False,
                                      prediction_loss_only=True,
                                      per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1, max_steps=10000, warmup_steps=4000,
                                      gradient_accumulation_steps=1,
                                      logging_first_step=True, max_grad_norm=1,
                                      save_total_limit=5, seed=0, dataloader_num_workers=4, no_cuda=not args.cuda)

    grid_size = (2,)
    dht = DHT(initial_peers=[args.init_peer], start=True, listen=False, max_workers=1, parallel_rpc=1, wait_timeout=0.1)

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

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    if args.restore_from_checkpoint is not None:
        checkpoint_path = Path(args.restore_from_checkpoint) / 'pytorch_model.bin'
        if checkpoint_path.exists():
            model_checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(model_checkpoint)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_wikitext['train'],
                      eval_dataset=tokenized_wikitext['validation'], tokenizer=tokenizer, data_collator=collator,
                      optimizers=(optimizer, lr_scheduler))

    print('Starting training')
    trainer.train(args.restore_from_checkpoint)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--hidden-dim', type=int, default=1024, required=False, help='main dimension for expert_cls')
    parser.add_argument('--init-peer')
    parser.add_argument('--restore-from-checkpoint')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    main(args)
