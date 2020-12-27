import os
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import datasets
import torch
import torch.nn as nn
from hivemind import RemoteMixtureOfExperts, DHT
from transformers import Trainer, TrainingArguments, EvaluationStrategy
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaConfig, MaskedLMOutput, RobertaLMHead
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast


class TrainerModel(nn.Module):
    def __init__(self, vocab_size, expert_dim, grid_size, dht, num_moe_blocks=12):
        super().__init__()
        self.config = RobertaConfig.from_pretrained('roberta-base')
        self.config.hidden_size = expert_dim
        self.config.vocab_size = vocab_size

        self.embeddings = RobertaEmbeddings(self.config)

        self.mixture = nn.ModuleList(
            [
                RemoteMixtureOfExperts(in_features=expert_dim, grid_size=grid_size, dht=dht,
                                       k_best=5, k_min=1, forward_timeout=2, backward_timeout=2,
                                       timeout_after_k_min=1,
                                       uid_prefix='expert.')
                for _ in range(num_moe_blocks)])

        self.lm_head = RobertaLMHead(self.config)

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
        embeddings = self.embeddings(input_ids)

        for layer in self.mixture:
            embeddings = layer(embeddings, src_key_padding_mask=~attention_mask.bool())

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
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

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

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)

    training_args = TrainingArguments(output_dir='mlm', overwrite_output_dir=True,
                                      do_train=True, do_eval=True, do_predict=False,
                                      evaluation_strategy=EvaluationStrategy.STEPS, prediction_loss_only=True,
                                      per_device_train_batch_size=32,
                                      per_device_eval_batch_size=32, max_steps=10000, warmup_steps=4000,
                                      gradient_accumulation_steps=8,
                                      logging_first_step=True,
                                      save_total_limit=5, seed=0, dataloader_num_workers=4, no_cuda=not args.cuda)

    grid_size = (16, 128)
    dht = DHT(initial_peers=[args.init_peer], start=True, listen=False, wait_timeout=1)

    model = TrainerModel(tokenizer.vocab_size, args.hidden_dim, grid_size, dht)

    if args.restore_from_checkpoint is not None:
        checkpoint_path = Path(args.restore_from_checkpoint) / 'pytorch_model.bin'
        if checkpoint_path.exists():
            model_checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(model_checkpoint)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_wikitext['train'],
                      eval_dataset=tokenized_wikitext['validation'], tokenizer=tokenizer, data_collator=collator)

    print('Starting training')
    trainer.train(args.restore_from_checkpoint)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--grid-size', '-g', type=int, default=100)
    parser.add_argument('--grid-dimensions', '-d', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=1024, required=False, help='main dimension for expert_cls')
    parser.add_argument('--init-peer')
    parser.add_argument('--restore-from-checkpoint')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    main(args)
