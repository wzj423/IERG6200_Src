from models.modeling_llama import LlamaPreTrainedModel,LlamaForCausalLM
from models.configuration_llama import LlamaConfig
from transformers import LlamaTokenizer
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling,TrainerCallback,TrainerControl,TrainerState,HfArgumentParser
from datasets import load_from_disk, load_dataset
import argparse
import math
from itertools import chain
import os
from dataclasses import dataclass,field
import torch
import numpy as np
from transformers import TrainerCallback, PreTrainedModel
from typing import Optional

# SEE https://huggingface.co/docs/transformers/main/main_classes/deepspeed#deepspeed-notebook
tokenizer_path = '/mnt/petrelfs/share_data/llama2_hf/llama-2-7b-hf/'
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
# tokenizer.model_max_length = 1024

def tokenize_function(examples):
    return tokenizer(examples['text'])

def group_texts(examples):
    max_seq_length=2048
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
    tensorboard_trace_handler: str = field(default="llama_trace", metadata={"help": "Name for the tensorboard trace."})



@dataclass
class DropTrainingArguments:
    """
    Arguments relating to Dropping.
    """
    drop_deep: bool = field(default=False, metadata={"help": "Random drop layers."})
    drop_weight: bool = field(default=False, metadata={"help": "Random drop K-V vectors."})




class BlockInfluenceLayerActivationCallback(TrainerCallback):
    def __init__(
        self,
        n_layers: int,
        interval_steps: int,
        model: PreTrainedModel,
        lisa_layers_attribute: Optional[str] = None,
        bi_warmup_steps: int = 0,  # BI预热步数
    ):
        super().__init__()
        self.n_layers = n_layers
        self.interval_steps = interval_steps
        self.model = model
        self.bi_warmup_steps = bi_warmup_steps

        # 确定访问层的方式
        class_to_layers_map = {
            "LlamaForCausalLM": "model.model.layers",
            "Qwen2ForCausalLM": "model.model.layers",
            "MistralForCausalLM": "model.model.layers",
            "MixtralForCausalLM": "model.model.layers",
            "GemmaForCausalLM": "model.model.layers",
            "GPT2LMHeadModel": "model.transformer.h",
            "HymbaForCausalLM": "model.model.layers",
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            assert lisa_layers_attribute is not None, "Please provide the attribute to access the layers of the model."
            self.layers_attribute = lisa_layers_attribute
        
        self.total_layers = len(eval("self." + self.layers_attribute))
        self.active_layers_indices = []
        self.bi_history = []  # 存储历史BI值用于平滑

    def freeze_all_layers(self):
        layers = eval("self." + self.layers_attribute)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_begin(self, args, state, control, **kwargs):
        # 在前向传播前启用BI计算
        if hasattr(self.model, 'set_compute_bi'):
            self.model.set_compute_bi(True)

    def on_step_end(self, args, state, control, **kwargs):
        # 检查是否需要切换激活层
        if state.global_step % self.interval_steps == 0:
            if state.global_step >= self.bi_warmup_steps:
                # 使用BI值选择层
                self.switch_active_layers_by_bi()
            else:
                # 预热期间使用随机选择
                self.switch_active_layers_random()
        
        # 关闭BI计算以节省计算资源
        if hasattr(self.model, 'set_compute_bi'):
            self.model.set_compute_bi(False)

    def switch_active_layers_random(self):
        """预热期间的随机层选择"""
        self.freeze_all_layers()
        layers = eval("self." + self.layers_attribute)
        self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
        print(f"[Warmup] Randomly activating layers at indices: {self.active_layers_indices}", flush=True)
        
        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True

    def switch_active_layers_by_bi(self):
        """基于BI值的层选择"""
        self.freeze_all_layers()
        
        # 获取最新的BI值
        if hasattr(self.model, 'get_block_influences'):
            current_bi = self.model.get_block_influences()
            if len(current_bi) > 0:
                # 添加到历史记录并进行平滑
                self.bi_history.append(current_bi)
                if len(self.bi_history) > 5:  # 保持最近5次的记录
                    self.bi_history.pop(0)
                
                # # 计算平滑后的BI值
                # if len(self.bi_history) > 1:
                #     smoothed_bi = np.mean(self.bi_history, axis=0)
                # else:
                #     smoothed_bi = current_bi
                
                # 选择BI值最大的n_layers层
                # 注意：BI值对应的是层与下一层之间的影响力，所以索引需要调整
                bi_indices = np.argsort(current_bi)[-self.n_layers:]  # 选择BI最大的层
                self.active_layers_indices = sorted(bi_indices)
                
                print(f"[BI-based] Current BI values: {[f'{bi:.4f}' for bi in current_bi]}", flush=True)
                # print(f"[BI-based] Smoothed BI values: {[f'{bi:.4f}' for bi in smoothed_bi]}", flush=True)
                print(f"[BI-based] Activating layers at indices: {self.active_layers_indices}", flush=True)
            else:
                # 如果无法获取BI值，回退到随机选择
                print("[Warning] No BI values available, falling back to random selection", flush=True)
                self.switch_active_layers_random()
                return
        else:
            print("[Warning] Model does not support BI calculation, falling back to random selection", flush=True)
            self.switch_active_layers_random()
            return
        
        # 激活选中的层
        layers = eval("self." + self.layers_attribute)
        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True

@dataclass
class LisaArguments:
    lisa: bool = field(default=True, metadata={"help": "Enable LISA training"})
    n_layers: int = field(default=8, metadata={"help": "Number of layers to activate"})
    interval_steps: int = field(default=20, metadata={"help": "Interval steps to switch active layers"})
    lisa_layers_attribute: Optional[str] = field(default=None, metadata={"help": "Attribute to access model layers"})
    # 新增BI相关参数
    bi_warmup_steps: int = field(default=0, metadata={"help": "Number of warmup steps before using BI-based selection"})
    # bi_smoothing_window: int = field(default=5, metadata={"help": "Window size for BI value smoothing"})

def main():
    parser = HfArgumentParser((ProfileArguments,LisaArguments,TrainingArguments)) #argparse.ArgumentParser()
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

    # config = LlamaConfig(drop_deep=drop_args.drop_deep,drop_weight=drop_args.drop_weight)
    # model = LlamaForCausalLM(config)

    config = LlamaConfig(vocab_size=32000,num_attention_heads=32,hidden_size=2048,intermediate_size=5632,max_position_embeddings=2048,\
                        num_hidden_layers=22, drop_deep=None,drop_weight=None)
    model = LlamaForCausalLM(config)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter number={count_parameters(model)}")
    # model = GPT2LMHeadModel.from_pretrained())
    
    # model.drop_weight = False
    # model.drop_deep=False

    

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # raw_datasets = load_dataset("openwebtext",cache_dir="/mnt/lustre/share_data/wuzijian/cache/cache/huggingface/datasets")
    # raw_datasets = raw_datasets['train'].train_test_split(train_size=10000,test_size=10000,shuffle=False,load_from_cache_file=True, train_indices_cache_file_name = "/mnt/lustre/share_data/wuzijian/cache/cache_train_split.arrow",test_indices_cache_file_name="/mnt/lustre/share_data/wuzijian/cache/cache_process_split.arrow")
    # # raw_datasets = load_dataset("RedPajama-Data-1T-Sample/RedPajama-Data-1T-Sample.py",trust_remote_code=True)
    # tokenized_datasets = raw_datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=64,
    #     remove_columns=raw_datasets["train"].column_names,
    #     desc="Running tokenizer on every text in dataset"
    #     ,cache_file_names={"train":"/mnt/lustre/share_data/wuzijian/cache_process_train0.arrow","test":"/mnt/lustre/share_data/wuzijian/cache_process_test0.arrow"},
    # )
    # tokenized_datasets.save_to_disk("./local_datasets0/")
    # # random_prompts = torch.randint(tokenizer.vocab_size, (100000, 256), device=torch.cuda.current_device())
    # tokenized_datasets = load_from_disk("./local_datasets0/")
    # tokenized_datasets = tokenized_datasets.map(
    #     group_texts,
    #     batched=True,
    #     num_proc=64,
    #     desc=f"Grouping texts in chunks of 2048",
    # )['train']

    # tokenized_datasets.save_to_disk("./local_datasets1/")
    #   # random_prompts = torch.randint(tokenizer.vocab_size, (100000, 256), device=torch.cuda.current_device())
    tokenized_datasets = load_from_disk("./local_datasets1/")
    # tokenized_datasets = load_dataset("togethercomputer/RedPajama-Data-1T-sample",trust_remote_code=True)

    print(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.train_test_split(train_size=2560,test_size=2560)

    # training_args = TrainingArguments(output_dir='gpt2_output',
    #                                 run_name='gpt2_normal',
    #                                 per_device_train_batch_size = 4,
    #                                 per_device_eval_batch_size = 4,
    #                                 num_train_epochs=1,
    #                                 save_steps = 1000,
    #                                 evaluation_strategy='steps',
    #                                 eval_steps = 1000,
    #                                 logging_steps=5,
    #                                 learning_rate=6e-4,
    #                                 gradient_accumulation_steps=1,
    #                                 save_safetensors=True,
    #                                 fp16=True,
    #                                 fp16_full_eval=True,
    #                                 deepspeed='ds_config_single_GPU_no_offload.json')

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA], 
                                schedule=torch.profiler.schedule(skip_first=5, wait=0, warmup=0, active=3, repeat=1),
                                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_args.tensorboard_trace_handler),
                                profile_memory=True,
                                with_stack=True,
                                record_shapes=True) as prof:
    
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
        trainer.add_callback(ProfilerCallback(prof=prof))
        # 替换原有的LayerActivationCallback为新的BlockInfluenceLayerActivationCallback
        if lisa_args.lisa:
            bi_callback = BlockInfluenceLayerActivationCallback(
                n_layers=lisa_args.n_layers,
                interval_steps=lisa_args.interval_steps,
                model=model,
                lisa_layers_attribute=lisa_args.lisa_layers_attribute,
                bi_warmup_steps=lisa_args.bi_warmup_steps if hasattr(lisa_args, 'bi_warmup_steps') else 100
            )
            trainer.add_callback(bi_callback)
            print(f"Added BlockInfluenceLayerActivationCallback with {lisa_args.n_layers} layers, "
                  f"interval {lisa_args.interval_steps} steps, warmup {getattr(lisa_args, 'bi_warmup_steps', 100)} steps")
        trainer.train()

if __name__ == '__main__':
    main()