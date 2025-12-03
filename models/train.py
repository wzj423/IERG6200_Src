from models.modeling_llama import  LlamaForCausalLM
from models.configuration_llama import LlamaConfig
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling,TrainerCallback,TrainerControl,TrainerState,HfArgumentParser
from datasets import load_from_disk, load_dataset
import argparse
import math
from itertools import chain
import os
from dataclasses import dataclass,field

tokenizer = AutoTokenizer.from_pretrained("./gpt2_tokenizer")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['text'])

def group_texts(examples):
    max_seq_length=1024
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // max_seq_length) * max_seq_length

    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

class ProfilerCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self, prof):
        self.prof = prof

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.prof.step()
        print("Profiler Step Added")


@dataclass
class ProfileArguments:
    """
    Arguments relating to Dropping.
    """
    tensorboard_trace_handler: str = field(default="gpt_trace_dropweight_DSz2_nooff_16g", metadata={"help": "Name for the tensorboard trace."})



@dataclass
class DropTrainingArguments:
    """
    Arguments relating to Dropping.
    """
    drop_deep: bool = field(default=False, metadata={"help": "Random drop layers."})
    drop_weight: bool = field(default=False, metadata={"help": "Random drop K-V vectors."})


def main():
    parser = HfArgumentParser((ProfileArguments,DropTrainingArguments,TrainingArguments))

    profile_args,drop_args,training_args = parser.parse_args_into_dataclasses()
    loss_fct = nn.CrossEntropyLoss()

    def preprocess_logits_for_metrics(logits, labels):
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, labels

    def compute_metrics(pred):
        loss = torch.from_numpy(pred.predictions[0])

        return {'perplexity': torch.exp(loss.mean())}

    config = LlamaConfig(vocab_size=50257,num_attention_heads=12,hidden_size=768,intermediate_size=4*768,max_position_embeddings=1024,\
                        num_hidden_layers=12, drop_deep=drop_args.drop_deep,drop_weight=drop_args.drop_weight)
    model = LlamaForCausalLM(config)


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    tokenized_datasets = load_from_disk("/code/remote/data/tokenized_openweb")
    tokenized_datasets = tokenized_datasets.train_test_split(train_size=1600,test_size=800)

    training_args = TrainingArguments(output_dir='gpt2_output',
                                    run_name='gpt2_normal',
                                    per_device_train_batch_size = 2,
                                    per_device_eval_batch_size = 2,
                                    num_train_epochs=1,
                                    save_steps = 1000,
                                    evaluation_strategy='steps',
                                    eval_steps = 1000,
                                    logging_steps=5,
                                    learning_rate=6e-4,
                                    gradient_accumulation_steps=32,
                                    save_safetensors=True,
                                    fp16=True,
                                    fp16_full_eval=True)

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
    #                                         torch.profiler.ProfilerActivity.CUDA], 
    #                             schedule=torch.profiler.schedule(skip_first=5, wait=0, warmup=0, active=3, repeat=1),
    #                             on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_args.tensorboard_trace_handler),
    #                             profile_memory=True,
    #                             with_stack=True,
    #                             record_shapes=True) as prof:
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    # trainer.add_callback(ProfilerCallback(prof=prof))
    trainer.train()

if __name__ == '__main__':
    main()