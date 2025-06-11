import json
import os
import sys
import importlib

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    else:
        dataset = Dataset.from_list(rows)
    
    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps
    
    # Differential Privacy configs
    dp_configs = {
        'no_dp': {'privacy_engine': None},
        'high_privacy': {'epsilon': 1.0, 'delta': 1e-5},
        'medium_privacy': {'epsilon': 3.0, 'delta': 1e-5},
        'low_privacy': {'epsilon': 8.0, 'delta': 1e-5}
    }
    dp_choice = training_cfg.dp_config if hasattr(training_cfg, 'dp_config') else 'no_dp'
    dp_params = dp_configs.get(dp_choice, dp_configs['no_dp'])

    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)

    # Attach Opacus PrivacyEngine if DP is enabled and loss is sft
    if dp_params['privacy_engine'] is None or training_cfg.loss != 'sft':
        print(f"Differential privacy: OFF (dp_config={dp_choice})")
    else:
        try:
            opacus = importlib.import_module('opacus')
            PrivacyEngine = opacus.PrivacyEngine
            print(f"Enabling Differential Privacy: {dp_choice} (epsilon={dp_params['epsilon']}, delta={dp_params['delta']})")
            privacy_engine = PrivacyEngine()
            trainer.optimizer, trainer.model, trainer.train_dataloader = privacy_engine.make_private_with_epsilon(
                module=trainer.model,
                optimizer=trainer.optimizer,
                data_loader=trainer.train_dataloader,
                epochs=training_cfg.epochs,
                target_epsilon=dp_params['epsilon'],
                target_delta=dp_params['delta'],
                max_grad_norm=1.0,
            )
        except ImportError:
            print("Opacus is not installed. Please install opacus to enable differential privacy training.")
        except Exception as e:
            print(f"Error initializing differential privacy: {e}")

    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg,finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.push_to_hub_merged(finetuned_model_id, tokenizer, save_method = "merged_16bit", token="HF_TOKEN", private=training_cfg.push_to_private)
    else:
        model.push_to_hub(finetuned_model_id, token="HF_TOKEN", private=training_cfg.push_to_private)
        tokenizer.push_to_hub(finetuned_model_id, token = "HF_TOKEN", private=training_cfg.push_to_private)


def main(config: str):
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])