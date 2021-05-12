from dataclasses import dataclass, field, asdict
from typing import Optional, List
from pathlib import Path
import os

from datasets import load_from_disk
from hivemind import DHT, CollaborativeOptimizer
from hivemind.utils import CompressionType, get_logger
from torch.optim.lr_scheduler import _LRScheduler
from torch_optimizer import Lamb
from transformers import get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, HfArgumentParser, \
    TrainingArguments, AlbertTokenizerFast, AlbertConfig
from transformers.optimization import get_linear_schedule_with_warmup

from modeling_albert_moe import AlbertForPreTraining
from trainer import NirvanaCheckpointTrainer

logger = get_logger(__name__)


class NoOpScheduler(_LRScheduler):
    """ Dummy scheduler for transformers.Trainer. The real scheduler is defined in CollaborativeOptimizer.scheduler """

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        logger.debug("Called NoOpScheduler.step")
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logger.debug("Called NoOpScheduler.load_state_dict")


@dataclass
class DatasetArguments:
    dataset_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    tokenizer_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    config_path: Optional[str] = field(default='.', metadata={"help": "Path to the dataset"})
    cache_dir: Optional[str] = field(default='.', metadata={"help": "Path to the cache"})


@dataclass
class BaseTrainingArguments:
    experiment_prefix: str = field(
        default='moe',
        metadata={"help": "A unique 'name' of this experiment, used to store metadata on the DHT"}
    )
    initial_peers: List[str] = field(
        default_factory=list,
        metadata={"help": "One or more peers (comma-separated) that will welcome you into the collaboration"}
    )
    dht_listen_on: str = field(
        default="[::]:*",
        metadata={"help": "Network interface used for incoming DHT communication. Default: all ipv6"}
    )


@dataclass
class AveragerArguments:
    averaging_expiration: float = field(
        default=5.0,
        metadata={"help": "Averaging group will wait for stragglers for at most this many seconds"}
    )
    averaging_timeout: float = field(
        default=30.0,
        metadata={"help": "Give up on averaging step after this many seconds"}
    )
    listen_on: str = field(
        default="[::]:*",
        metadata={"help": "Network interface used for incoming averager communication. Default: all ipv6"}
    )
    min_refresh_period: float = field(
        default=1,
        metadata={"help": "Wait for at least this many seconds before fetching new collaboration state"}
    )
    max_refresh_period: float = field(
        default=600,
        metadata={"help": "Wait for at most this many seconds before fetching new collaboration state"}
    )
    default_refresh_period: float = field(
        default=300,
        metadata={"help": "Attempt to fetch collaboration state every this often until successful"}
    )
    expected_drift_peers: float = field(
        default=3,
        metadata={"help": "Trainer assumes that this many new peers can join per step"}
    )
    expected_drift_rate: float = field(
        default=0.2,
        metadata={"help": "Trainer assumes that this fraction of current size can join per step"}
    )
    performance_ema_alpha: float = field(
        default=0.1,
        metadata={"help": "Uses this alpha for moving average estimate of samples per second"}
    )
    target_group_size: int = field(
        default=256,
        metadata={"help": "Maximum group size for all-reduce"}
    )
    metadata_expiration: float = field(
        default=30,
        metadata={"help": "Peer's metadata will be removed if not updated in this many seconds"}
    )


@dataclass
class CollaborativeOptimizerArguments:
    target_batch_size: int = field(
        default=4096,
        metadata={"help": "Perform optimizer step after all peers collectively accumulate this many samples"}
    )
    client_mode: bool = field(
        default=False,
        metadata={"help": "Of True, runs training without incoming connections, in a firewall-compatible mode"}
    )
    batch_size_lead: int = field(
        default=0,
        metadata={"help": "Optional: begin looking for group in advance, this many samples before target_batch_size"}
    )
    bandwidth: float = field(
        default=100.0,
        metadata={"help": "Available network bandwidth, in mbps (used for load balancing in all-reduce)"}
    )
    compression: str = field(
        default="FLOAT16",
        metadata={"help": "Use this compression when averaging parameters/gradients"}
    )


@dataclass
class CollaborationArguments(AveragerArguments, CollaborativeOptimizerArguments, BaseTrainingArguments):
    statistics_expiration: float = field(
        default=600,
        metadata={"help": "Statistics will be removed if not updated in this many seconds"}
    )
    endpoint: Optional[str] = field(
        default=None,
        metadata={"help": "This node's IP for inbound connections, used when running from behind a proxy"}
    )


@dataclass
class MixtureTrainingArguments(TrainingArguments):
    balancing_loss_weight: Optional[float] = field(default=1e-2)


def main(dataset_args, training_args, collaboration_args, args):
    tokenizer = AlbertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)

    tokenized_datasets = load_from_disk(dataset_args.dataset_path)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=not args.lm)

    collaboration_args_dict = asdict(collaboration_args)
    collaboration_args_dict.pop('statistics_expiration')

    grid_size = (args.grid_size,)
    dht = DHT(initial_peers=collaboration_args_dict.pop('initial_peers'),
              listen=not collaboration_args_dict['client_mode'],
              listen_on=collaboration_args_dict.pop('dht_listen_on'),
              endpoint=collaboration_args_dict.pop('endpoint'), start=True, max_workers=256, wait_timeout=2)

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

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    adjusted_target_batch_size = collaboration_args_dict.pop('target_batch_size') \
                                 - collaboration_args_dict.pop('batch_size_lead')

    collaborative_optimizer = CollaborativeOptimizer(
        opt=optimizer, dht=dht, scheduler=lr_scheduler, prefix=collaboration_args_dict.pop('experiment_prefix'),
        compression_type=CompressionType.Value(collaboration_args_dict.pop('compression')),
        batch_size_per_step=total_batch_size_per_step, throughput=collaboration_args_dict.pop('bandwidth'),
        target_batch_size=adjusted_target_batch_size, client_mode=collaboration_args_dict.pop('client_mode'),
        verbose=True, start=True, **collaboration_args_dict
    )

    trainer = NirvanaCheckpointTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(collaborative_optimizer, NoOpScheduler(collaborative_optimizer))
    )

    trainer.train(resume_from_checkpoint=latest_checkpoint_dir)


if __name__ == '__main__':
    parser = HfArgumentParser((DatasetArguments, MixtureTrainingArguments, CollaborationArguments))
    parser.add_argument('--grid-size', type=int, required=True)
    parser.add_argument('--lm', action='store_true')

    dataset_args, training_args, collaboration_args, args = parser.parse_args_into_dataclasses()
    main(dataset_args, training_args, collaboration_args, args)
