import torch
from torch import nn

import numpy as np
import pandas
import os
import shutil
import copy
from copy import deepcopy

import json

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, load_dataset

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import textattack
import transformers

import huggingface_hub, os

import matplotlib.pyplot as plt
from tqdm import tqdm

from langdetect.detector import LangDetectException
from langdetect import detect

import itertools
import pandas as pd

from datasets import DatasetDict, concatenate_datasets

from attack import A2TYoo2021

import shutil, pickle
import time

import warnings

import argparse

from sklearn.model_selection import train_test_split
from datasets import Dataset

import wandb

import psutil

import time
from threading import Thread

import pynvml

def log_system_metrics():
    ram_usage = psutil.virtual_memory().percent  # Get RAM usage in percentage
    wandb.log({"RAM Usage (%)": ram_usage})


def log_gpu_metrics():
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_usage = (mem_info.used / mem_info.total) * 100  # GPU memory usage in percentage
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # GPU utilization in percentage
        wandb.log({f"GPU {i} Memory Usage (%)": gpu_mem_usage, f"GPU {i} Utilization (%)": gpu_util})


class CFG:
    val_split = 0.1
    test_split = 0.1
    models = ['google-bert/bert-base-uncased', 'microsoft/MiniLM-L12-H384-uncased', 'google/electra-small-discriminator']
    models_dir = 'models'
    logs_dir = 'logs'
    data_dir = 'datasets'
    base_dir = '/'
    target_dir = '/'
    max_max_rows = 18000
    max_rows = 18000
    max_num_epochs = 3
    max_attack_ex = 512
    
class ExperimentLogger:
  def __init__(self, base_directory, datasets):
    self.base_directory = base_directory

    self.models_directory = os.path.join(base_directory, 'models')
    self.logs_directory = os.path.join(base_directory, 'logs')
    self.tmp_directory = os.path.join(base_directory, 'tmp')

    self.dirs = [self.models_directory, self.logs_directory, self.tmp_directory]
    self.datasets = datasets

    self.models_paths = {}
    self.model_results = {}
    self.attack_results = {}

  def get_attack_key(self, model_name, attack_name, dataset_id, part='train'):
    return (model_name, self.datasets[dataset_id][0], attack_name, part)

  def add_attack_result(self, model_name, attack_name, dataset_id, result, part='train'):
    print(f'Logged: \n Attack result \n {model_name} \n {self.datasets[dataset_id][0]} \n {part} \n {result} \n')
    self.attack_results[(model_name, self.datasets[dataset_id][0], attack_name, part)] = result

  def add_model_result(self, model_name, dataset_id, result, part='train'):
    print(f'Logged: \n Model result \n {model_name} \n {self.datasets[dataset_id][0]} \n {part} \n {result} \n')
    self.model_results[(model_name, self.datasets[dataset_id][0], part)] = result

  def add_model_path(self, model_name, result):
    print(f'Logged: \n Model path \n {model_name} \n {result} \n')
    self.models_paths[model_name] = result
    
class Experiment:
  def __init__(self, base_directory, datasets, model_name, run, seed=42):
    self.base_directory = base_directory
    print(f'Base directory: {base_directory}')

    self.models_directory = os.path.join(base_directory, 'models')
    self.logs_directory = os.path.join(base_directory, 'logs')
    self.tmp_directory = os.path.join(base_directory, 'tmp')
    self.dirs = [self.models_directory, self.logs_directory, self.tmp_directory]

    self.datasets = datasets
    self.model_name = model_name

    self.exp_logger = ExperimentLogger(base_directory, datasets)
    self.seed = seed
    self.run = run

  def get_preprocess_function(self, tokenizer, max_length=256):  
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    return preprocess_function


  def _generate_sequences(self, m):
    return list(itertools.combinations(range(len(self.datasets)), m))

  @classmethod
  def compute_metrics(cls, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

  def _get_model_name(self, datasets, train_methods, epochs, run):
    sqns = []
    for dt, tm, ep in zip(datasets, train_methods, epochs):
      sqns.append(f'd_{dt}_e_{ep}_t_{tm}_r_{run}')
    print(sqns)

    return '|'.join([self.model_name, '|'.join(sqns)])

  def _ensure_folder_structure(self):
    for dir in self.dirs:
      if not os.path.exists(dir):
        os.makedirs(dir)

  def train_model(self, model, model_name, tokenizer, dataset, epochs=4, load_best_model_at_end=True):
    print(f'Model: {model_name} Started training:')
    print('<' * 40 + '\n')

    tokenizer = tokenizer
    tokenized_dataset = dataset.map(self.get_preprocess_function(tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        seed=self.seed,
        output_dir=self.tmp_directory,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.02,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=load_best_model_at_end,
        push_to_hub=False,
        metric_for_best_model='accuracy',
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=Experiment.compute_metrics,
    )

    trainer.train()

    path_to_save = os.path.join(self.models_directory, model_name)
    print(f'Saving path: {path_to_save}')

    if not os.path.exists(path_to_save):
      os.makedirs(path_to_save)

    model.save_pretrained(path_to_save)
    self.exp_logger.add_model_path(model_name, path_to_save)
    print(f'Finished training:')
    print('>' * 40 + '\n')
    return model

  def evaluate_model(self, model, model_name, tokenizer, dataset):
    print(f'Model: {model_name} Started evaluating:')
    print('<' * 40 + '\n')

    tokenizer = tokenizer
    tokenized_dataset = dataset.map(self.get_preprocess_function(tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=self.tmp_directory,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=False,
        metric_for_best_model='accuracy',
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=Experiment.compute_metrics,
    )

    eval_result = trainer.evaluate(tokenized_dataset["test"])
    wandb.log(eval_result)

    print(f'Finished evaluating: \n')

    print(f'Metric results: {eval_result} \n ')
    print('>' * 40 + '\n')
    return eval_result

class CustomAttackerCl:
  def __init__(self, attack_method=A2TYoo2021, outdir='/content/tmp'):
    self.attack_method = attack_method
    self.name = 'A2TYoo2021'
    self.outdir = outdir

  def compute_stats(self, out_dir):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(out_dir)

    # Extract columns into numpy arrays
    ground_truth_output = np.array(df['ground_truth_output'])
    original_output = np.array(df['original_output'])
    perturbed_output = np.array(df['perturbed_output'])

    # Compute accuracy for original and perturbed outputs
    original_accuracy = accuracy_score(ground_truth_output, original_output)
    perturbed_accuracy = accuracy_score(ground_truth_output, perturbed_output)

    # Compute precision, recall, and F1 score for original outputs
    original_precision, original_recall, original_f1, _ = precision_recall_fscore_support(ground_truth_output, original_output, average='binary')

    # Compute precision, recall, and F1 score for perturbed outputs
    perturbed_precision, perturbed_recall, perturbed_f1, _ = precision_recall_fscore_support(ground_truth_output, perturbed_output, average='binary')

    # Calculate attack success rate
    attack_success_rate = len(df[df.result_type == 'Successful']) / len(df[df.result_type != 'Skipped'])

    mean_queries = np.mean(df.num_queries)

    # Return all computed metrics in a dictionary
    return {
        'original_accuracy': original_accuracy,
        'perturbed_accuracy': perturbed_accuracy,
        'original_precision': original_precision,
        'original_recall': original_recall,
        'original_f1': original_f1,
        'perturbed_precision': perturbed_precision,
        'perturbed_recall': perturbed_recall,
        'perturbed_f1': perturbed_f1,
        'attack_success_rate': attack_success_rate,
        'mean_queries': mean_queries
    }

  def _get_name(self):
    return self.name

  def build_a2t_attack(self, model_name, model_wrapper, dataset, dataset_name, n_ex):
    attack = self.attack_method.build(model_wrapper, mlm=False)
    #num_examples = n_ex
    # Set up file naming
    #name = f'a2t_attack\|n_{num_examples}\|m_{model_name}\|d_{dataset_name}'
    name = 'attack_res'
    out_dir = os.path.join(self.outdir, name + '.csv')

    # Configure the attack arguments
    #print('heheheheheh'*20)
    attack_args = textattack.AttackArgs(num_examples=n_ex, log_to_csv=out_dir,
                                        checkpoint_interval=None, disable_stdout=True,
                                        num_workers_per_device=1, query_budget=80,
                                        parallel=True)

    # Construct the attacker
    attacker = textattack.Attacker(attack, dataset, attack_args)

    return attacker, out_dir

  def attack(self, model, model_name, tokenizer, dataset, dataset_name, max_attack_ex=CFG.max_attack_ex):
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    n_ex = min(max_attack_ex, len(dataset))

    dataset_small = dataset.select(range(n_ex))
    dataset_small = textattack.datasets.HuggingFaceDataset(dataset_small)


    attacker, out_dir_file = self.build_a2t_attack(model_name, model_wrapper, dataset_small, dataset_name, n_ex)
    attacker.attack_dataset()
    results = self.compute_stats(out_dir_file)
    wandb.log(results)

    del attacker
    del dataset_small

    return results, self._get_name()


#@title Experiment: Basic experiment

class BasicCLExperiment(Experiment):
  def __init__(self, base_directory, datasets, model_name, seed=42, training_method=['u', 'u'], epochs=[1, 1], base_epochs=4, base_len=1,  load_best_model_at_end=False, max_attack_ex=1024, run=0):
    super().__init__(base_directory, datasets, model_name, run, seed)
    self.base_epochs = base_epochs
    self.sequences = self._generate_sequences(base_len)

    self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    #self.model_paths = {}
    self.training_method = training_method
    self.load_best_model_at_end = load_best_model_at_end
    self.max_attack_ex = max_attack_ex
    self.epochs = epochs
    self.seed = seed
    self.run = run

  def attack_model(self, model, model_name, tokenizer, dataset, dataset_name, outdir):
    cust_attacker = CustomAttackerCl(outdir=outdir)
    dataset = dataset['test']
    results, name = cust_attacker.attack(model, model_name, tokenizer, dataset, dataset_name, max_attack_ex=self.max_attack_ex)
    return results, name

  def _generate_sequences(self, m):
    return list(itertools.permutations(range(len(self.datasets)), m))

  def run_experiment(self):
    self._ensure_folder_structure()

    for sqn in self.sequences:
      model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2, ignore_mismatched_sizes=True,
      )

      for i, dataset_idx in enumerate(sqn):
        dataset_name = self.datasets[dataset_idx][0]
        dataset = self.datasets[dataset_idx][1]
        print(i)

        model_name = self._get_model_name(sqn[:i + 1], self.training_method[:i + 1], self.epochs[:i + 1], self.run)#self._get_model_name(sqn, i)

        print(model_name)

        if model_name in self.exp_logger.models_paths:
          print('-'*20 + 'Decision: Loading model \n')
          abs_path = os.path.join(self.models_directory, model_name)
          del model

          model = AutoModelForSequenceClassification.from_pretrained(
            abs_path, num_labels=2, ignore_mismatched_sizes=True
          )
        else:
          print('-'*20 + 'Decision: Training model \n')
          model = self.train_model(model, model_name, self.tokenizer, dataset, epochs=self.epochs[i], load_best_model_at_end=self.load_best_model_at_end)

          print('-'*20 + f'Decision: Evaluating model on end {dataset_name} dataset: \n')
          results = self.evaluate_model(model, model_name, self.tokenizer, dataset)
          self.exp_logger.add_model_result(model_name, dataset_idx, results, 'test')

        cust_attacker = CustomAttackerCl()
        attack_name = cust_attacker._get_name()
        att_key = self.exp_logger.get_attack_key(model_name, attack_name, dataset_idx)

        if att_key not in self.exp_logger.attack_results:

          print('-'*20 + f'Decision: Attacking model on end {dataset_name} dataset: \n')
          result, attack_name = self.attack_model(model, model_name, self.tokenizer, dataset, dataset_name, os.path.join(self.base_directory, 'tmp'))
          #self.exp_logger.add_model_result(model_name, dataset_idx, results, 'test')
          self.exp_logger.add_attack_result(model_name, attack_name, dataset_idx, result)
        else:
          print('-'*20 + f'Decision: SKIPPING Attacking model on end {dataset_name} dataset: \n')
          
def set_configs():
    access_token = "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"
    wandb_api_key = '3f6581f87b267c4d2f5d0cac29894fcee4a9fc8d'

    huggingface_hub.login(access_token)
    os.environ['HF_AUTH_TOKEN'] = access_token
    os.environ['WANDB_API_KEY'] = wandb_api_key

def split_dataset(dataset):
    train_testvalid = dataset.train_test_split(0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    ds = DatasetDict({
      'train': train_testvalid['train'],
      'test': test_valid['test'],
      'valid': test_valid['train']})
    return ds
  
def split_and_shuffle_dataset(dataset, stratify_col_name='label', seed=42):
    df = dataset.to_pandas()

    train_df, test_valid_df = train_test_split(df, test_size=0.2, stratify=df[stratify_col_name], random_state=seed)

    test_df, valid_df = train_test_split(test_valid_df, test_size=0.5, stratify=test_valid_df[stratify_col_name], random_state=seed)

    train_ds = Dataset.from_pandas(train_df).shuffle(seed=seed)
    test_ds = Dataset.from_pandas(test_df).shuffle(seed=seed)
    valid_ds = Dataset.from_pandas(valid_df).shuffle(seed=seed)

    ds_dict = DatasetDict({
        'train': train_ds,
        'test': test_ds,
        'valid': valid_ds
    })

    return ds_dict
    
def load_datasets(max_ex_len, seed=42):
    dataset_linguistic_bias = load_dataset("BogdanTurbal/rai_linguistic_bias_f").shuffle(seed=seed)
    dataset_gender_bias = load_dataset("BogdanTurbal/rai_gender_bias_f").shuffle(seed=seed)
    dataset_hate_speech = load_dataset("BogdanTurbal/rai_hate_speech_f").shuffle(seed=seed)
    
    if max_ex_len != 0:
        dataset_linguistic_bias['train'] = dataset_linguistic_bias['train'].select(range(max_ex_len))
        dataset_gender_bias['train'] = dataset_gender_bias['train'].select(range(max_ex_len))
        dataset_hate_speech['train'] = dataset_hate_speech['train'].select(range(max_ex_len))
        

    dataset_linguistic_bias = split_and_shuffle_dataset(dataset_linguistic_bias['train'])
    dataset_gender_bias = split_and_shuffle_dataset(dataset_gender_bias['train'])
    dataset_hate_speech = split_and_shuffle_dataset(dataset_hate_speech['train'])

    datasets = [('rai_linguistic_bias', dataset_linguistic_bias), ('rai_gender_bias', dataset_gender_bias), ('rai_hate_speech', dataset_hate_speech)]
    
    return datasets
    
def args_parser():
    parser = argparse.ArgumentParser(description="Argument parser")
    
    parser.add_argument("cr_d", help="Current dir")
    parser.add_argument("sr_d", help="Save dir")
    parser.add_argument("--msl", help="Sqn len", default=2, type=int)
    parser.add_argument("--ne", help="Num epochs", default=2, type=int)
    parser.add_argument("--mae", help="Max attack examples", default=1024, type=int)
    parser.add_argument("--mel", help="Max num of examples in dataset", default=0, type=int)
    parser.add_argument("--seed", help="Seed", default=42, type=int)
    parser.add_argument("--run", help="Run", default=0, type=int)
    parser.add_argument("--mod_id", help="Model id", default=0, type=int)
    parser.add_argument("--load_best", help="Load best epoch", default=0, type=int)
    
    return parser.parse_args()
    #parser.add_argument("--optional-arg", help="Optional argument", default="default value")
    

def save_data(current_dir, exp, model, run):
  model_n = model.split('/')[-1]
  with open(f'{current_dir}models_paths_{model_n}_{run}.pkl', 'wb') as file:
    pickle.dump(exp.exp_logger.models_paths, file)

  with open(f'{current_dir}model_results_{model_n}_{run}.pkl', 'wb') as file:
    pickle.dump(exp.exp_logger.model_results, file)

  with open(f'{current_dir}attack_results_{model_n}_{run}.pkl', 'wb') as file:
    pickle.dump(exp.exp_logger.attack_results, file)

def init_configs(base_epochs, model_name, training_method, max_attack_ex, max_ex, run, seed):
  pynvml.nvmlInit()
  wandb.login(key="3f6581f87b267c4d2f5d0cac29894fcee4a9fc8d")
  
  wandb.init(project="rai_trans_experiment", config={
        "epochs": base_epochs,
        "model_name": model_name,
        "training_method": training_method,
        "max_attack_ex": max_attack_ex,
        "max_ex": max_ex, 
        "run": run,
        "seed": seed
    })

def monitor_resources():
    while True:
        log_system_metrics()
        log_gpu_metrics()
        time.sleep(60)

def main():
    set_configs()
    args = args_parser()
    current_dir = args.cr_d
    save_dir = args.sr_d
    num_epochs = args.ne
    max_len = args.msl
    max_attack_ex = args.mae
    max_examples_num = args.mel
    seed = args.seed
    run = args.run
    model_id = args.mod_id
    load_best_model_at_end = (args.load_best == 1)
    
    init_configs(num_epochs, CFG.models[model_id], 'u', max_attack_ex, max_examples_num, run, seed)
    monitor_thread = Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    datasets = load_datasets(max_examples_num)
    
    print("Used datasets:")
    print(datasets)
    
    #warnings.filterwarnings("ignore")
    #seeds = [1, 42, 1234]
     
    for model in [CFG.models[model_id]]:
        exp = BasicCLExperiment(current_dir, datasets, model, seed=seed, base_epochs=num_epochs, epochs=[num_epochs] * max_len, training_method=['u'] * max_len, max_attack_ex=max_attack_ex, base_len=max_len, run=run, load_best_model_at_end=load_best_model_at_end)
        exp.run_experiment()

        save_data(save_dir, exp, model, run)

if __name__ == "__main__":
    main()