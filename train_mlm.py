from argparse import ArgumentParser
from multiprocessing import cpu_count

import datasets
import torch.nn as nn
from hivemind import RemoteMixtureOfExperts, DHT
from transformers import Trainer, TrainingArguments, EvaluationStrategy
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.modeling_roberta import RobertaEmbeddings, RobertaConfig, MaskedLMOutput
from transformers.tokenization_roberta_fast import RobertaTokenizerFast


class TrainerModel(nn.Module):
    def __init__(self, vocab_size, expert_dim, grid_size, dht, num_moe_blocks=2):
        super().__init__()
        self.config = RobertaConfig.from_pretrained('roberta-base')
        self.config.hidden_size = expert_dim
        self.config.vocab_size = vocab_size

        self.embeddings = RobertaEmbeddings(self.config)

        self.mixture = nn.Sequential(
            *[RemoteMixtureOfExperts(in_features=expert_dim, grid_size=grid_size, dht=dht,
                                     k_best=5, k_min=1, forward_timeout=10, backward_timeout=1, timeout_after_k_min=1,
                                     uid_prefix='expert.')
              for _ in range(num_moe_blocks)])

        self.lm_head = nn.Linear(expert_dim, vocab_size)

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
        output = self.mixture(embeddings)
        prediction_scores = self.lm_head(output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = nn.functional.cross_entropy(prediction_scores.view(-1, self.config.vocab_size),
                                                         labels.view(-1))

        if not return_dict:
            output = (prediction_scores, output)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
        )


def main(args):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
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
        load_from_cache_file=True,
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)

    training_args = TrainingArguments(output_dir='mlm', overwrite_output_dir=True, do_train=True, do_eval=True,
                                      do_predict=False,
                                      evaluation_strategy=EvaluationStrategy.STEPS, prediction_loss_only=True,
                                      per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1, max_steps=10000, warmup_steps=4000,
                                      gradient_accumulation_steps=1,
                                      logging_first_step=True,
                                      save_total_limit=5, seed=0, dataloader_num_workers=1)

    grid_size = tuple(args.grid_size for _ in range(args.grid_dimensions))
    dht = DHT(initial_peers=[args.init_peer], start=True, listen=False, wait_timeout=5)

    model = TrainerModel(tokenizer.vocab_size, args.hidden_dim, grid_size, dht)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_wikitext['train'],
                      eval_dataset=tokenized_wikitext['validation'], tokenizer=tokenizer, data_collator=collator)

    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--grid-size', '-g', type=int, default=100)
    parser.add_argument('--grid-dimensions', '-d', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256, required=False, help='main dimension for expert_cls')
    parser.add_argument('--init-peer')
    args = parser.parse_args()
    main(args)
