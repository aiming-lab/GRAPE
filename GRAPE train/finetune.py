import torch.nn.functional as F
import torch
import tqdm
from PIL import Image 
from torch.utils.data import IterableDataset
import random
from dataclasses import dataclass
import numpy as np
from transformers import PreTrainedTokenizerBase
from pathlib import Path
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, Type
import os
from collections import deque
#import time
import draccus
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from accelerate import PartialState
from prismatic.vla.action_tokenizer import ActionTokenizer
import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from datasets import disable_caching
disable_caching()
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
 
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
   
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
       # print(prompt_builder.get_prompt())

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!

        return dict(instruction=lang, action=self.action_tokenizer(action), image=img,action_num=action)

class RLDSDataset(IterableDataset):

    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        #batch_transform: None,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 0,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix = data_root_dir, data_mix


        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield rlds_batch

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

def window_batch(Episode_dataset):

    all_batches = []

    window_size=8

    for item in tqdm.tqdm(Episode_dataset):
        
        images = np.squeeze(item['observation']['image_primary'])
        instructions = item['task']['language_instruction']
        actions = np.squeeze(item['action'])

        num_windows = len(images) - window_size + 1

        item_batches = []

        for i in range(num_windows):
            window_images = images[i:i+window_size]
            window_instructions = instructions[i:i+window_size]
            window_actions = actions[i:i+window_size]

            prompts = []
            for step in range(window_size):
                prompt = {
                    'instruction': window_instructions[step],
                    'image': Image.fromarray(window_images[step]),
                    'action': window_actions[step]
                }
                prompts.append(prompt)
            item_batches.append(prompts)
        all_batches.append(item_batches)

    print("Total number of items:", len(all_batches))

    first_item_batches = all_batches[0]
    first_batch = first_item_batches[0]
    first_step = first_batch[0]
    print("First Step in First Batch of First Item:")
    print("Instruction:", first_step['instruction'])
    print("Image Shape:", first_step['image'])
    print("Action:", first_step['action'])
    return all_batches

class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        #batch_transform: None,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 0,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix = data_root_dir, data_mix
        self.batch_transform=batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield rlds_batch

def transform(processor,data):
    IGNORE_INDEX=-100
    action_tokenizer=ActionTokenizer(processor.tokenizer)
    action=data['action']
    action_description=action_tokenizer(action)
    tokenizer=processor.tokenizer
    instruction=str(data['instruction'])[2:-1]
    lang=f"In: What action should the robot take to {instruction}?\nOut: {action_description}</s>"
    input_ids = tokenizer(lang, add_special_tokens=True).input_ids
    labels = list(input_ids)
    img=data['image']
    input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    labels[: -(len(action) + 1)] = IGNORE_INDEX
    input_ids=torch.unsqueeze(input_ids,dim=0)
    labels=torch.unsqueeze(labels,dim=0)
    attention_mask = input_ids.ne(32000)
    pixel_values = processor.image_processor.apply_transform(img)
    pixel_values=torch.unsqueeze(pixel_values,dim=0)
    return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels,attention_mask=attention_mask,dataset_name="success")

class TrajDataset(IterableDataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __iter__(self):
        
        for item in self.data_list:
            yield item

def basic_collate_fn(batch):
    return batch

def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
    if not is_encoder_decoder:                                                                                     
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    device_id
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    loss_type="sigmoid"
    beta=0.1
    chosen_logratios = policy_chosen_logps.to(device_id) -  reference_chosen_logps.to(device_id)
    rejected_logratios = policy_rejected_logps.to(device_id) -reference_rejected_logps.to(device_id)

    pi_logratios = policy_chosen_logps - policy_rejected_logps

    ref_logratios = reference_chosen_logps - reference_rejected_logps

    pi_logratios = pi_logratios.to(device_id)
    ref_logratios = ref_logratios.to(device_id)
    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits)
        )
        # print("beta:",beta,"loss:",losses)
    elif loss_type == "mixture":
        losses = (
            -F.logsigmoid(beta * logits) 
        )

    chosen_rewards = (
        beta
        * (
            policy_chosen_logps.to(device_id) - reference_chosen_logps.to(device_id)
        ).detach()
    )
    rejected_rewards = (
        beta
        * (
            policy_rejected_logps.to(device_id)
            - reference_rejected_logps.to(device_id)
        ).detach()
    )

    return losses, chosen_rewards, rejected_rewards

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "path/to/sft_model"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    #data_root_dir: Path = Path("/data/zhaoyang_wang/projects/OpenVLA/dataset")        # Path to Open-X dataset directory
    chosen_traj_dir: Path = Path("path/to/chosen/traj")
    rejected_traj_dir: Path = Path("path/to/rejected/traj")
    dataset_name: str = "bridge_orig"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 1                                            # Fine-tuning batch size
    max_steps: int = 20000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100000                           # Dataloader shuffle buffer size (can reduce if OOM)
    epoch: int = 10
    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "your project"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "your entity"                          # Name of entity to log under

    # fmt: on
def flatshape(traj_a,traj_b):
    flat_traj_a = [step for window in traj_a for step in window]
    flat_traj_b = [step for window in traj_b for step in window]

    # 2. 对展平后的列表进行 shuffle（两者保持相同的随机顺序）
    combined = list(zip(flat_traj_a, flat_traj_b))  # 将两个列表打包在一起，保持同步打乱
    random.shuffle(combined)  # 随机打乱
    flat_traj_a_shuffled, flat_traj_b_shuffled = zip(*combined)  # 解包打乱后的列表

    # 3. 将打乱后的二维列表还原为三维 (30, traj_num, 8)
    traj_num_a = len(traj_a[0])  # 获取原始 traj_num
    traj_a_shuffled = [flat_traj_a_shuffled[i:i+traj_num_a] for i in range(0, len(flat_traj_a_shuffled), traj_num_a)]
    traj_b_shuffled = [flat_traj_b_shuffled[i:i+traj_num_a] for i in range(0, len(flat_traj_b_shuffled), traj_num_a)]

    return traj_a_shuffled,traj_b_shuffled



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    print(device_id)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=None
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 32),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()


    # Load reference model(OpenVLA-SFT)
    vla_ref = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=None
)
    vla_ref = vla_ref.to(device_id)
    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    vla_ref= DDP(vla_ref, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    RLDSTransform=RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    episode_chosen=EpisodicRLDSDataset(
        cfg.chosen_traj_dir,
        cfg.dataset_name,
        resize_resolution=tuple([224,224]),
        batch_transform=RLDSTransform,
        shuffle_buffer_size=1,
        image_aug=False,
    )

    episode_rejected=EpisodicRLDSDataset(
        cfg.rejected_traj_dir,
        cfg.dataset_name,
        resize_resolution=tuple([224,224]),
        batch_transform=RLDSTransform,
        shuffle_buffer_size=1,
        image_aug=False,
    )

    data_length=episode_chosen.__len__()
    print("Data Length:",data_length)

    episode_chosen_iter=iter(episode_chosen)
    episode_rejected_iter=iter(episode_rejected)

    episode_chosen_list=[]
    episode_rejected_list=[]
    # Load trajectory data from rlds files and then convert it to list
    for i in range(data_length-7):

        item_s=next(episode_chosen_iter)
        item_f=next(episode_rejected_iter)

        episode_chosen_list.append(item_s)
        episode_rejected_list.append(item_f)
    
    # Using a sliding window of size 8, the list is converted into a batch one by one.
    chosen_batch=window_batch(episode_chosen_list)
    rejected_batch=window_batch(episode_rejected_list)

    # Flat the batch and then shuffle it 
    chosen_batch,rejected_batch=flatshape(chosen_batch,rejected_batch)

    # Wrap it as iterable dataset
    traj_dataset_success=TrajDataset(chosen_batch)
    traj_dataset_fail=TrajDataset(rejected_batch)

    collator=basic_collate_fn

    # Wrap dataset by Dataloader
    dataloader_chosen=DataLoader(
        traj_dataset_success,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    dataloader_rejected=DataLoader(
        traj_dataset_fail,
        batch_size=1,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_acc=deque(maxlen=cfg.grad_accumulation_steps)
    

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        traj_count=0
        vla.train()
        for epoch in range(cfg.epoch):
            for batch_idx, (batch_chosen, batch_rejected) in tqdm.tqdm(enumerate(zip(dataloader_chosen, dataloader_rejected))):

                optimizer.zero_grad()
                batch_chosen_list=list(batch_chosen)[0]
                batch_rejected_list=list(batch_rejected)[0]

                traj_num=min(len(list(batch_chosen)[0]),len(list(batch_rejected)[0]))
                
                for traj_idx in range(traj_num):
                    chosen_rewards_sum=0
                    rejected_rewards_sum=0
                    logps_chosen=0
                    logps_rejected=0
                    traj_count+=1
                    loss_sum=0
                    optimizer.zero_grad()
                    # Calculate traj-loss in each batch
                    for step in range(8): 

                        data_chosen=batch_chosen_list[traj_idx][step]
                        data_rejected=batch_rejected_list[traj_idx][step]  

                        data_chosen=transform(processor,data_chosen)
                        data_rejected=transform(processor,data_rejected)

                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            # Calculate chosen_policy likelihood
                            output_chosen_policy: CausalLMOutputWithPast = vla(
                                input_ids=data_chosen["input_ids"].to(device_id),
                                attention_mask=data_chosen["attention_mask"].to(device_id),
                                pixel_values=data_chosen["pixel_values"].to(torch.bfloat16).to(device_id),
                                labels=data_chosen["labels"].to(device_id),
                            )
                            labels=data_chosen["labels"]
                            labels=labels.to(device_id)
                            logits=output_chosen_policy.logits
                            #project patch labels in language labels
                            projected_patch_labels = torch.full(
                                (labels.shape[0], 256),
                                fill_value=-100,
                                dtype=labels.dtype,
                                device=device_id,
                            )

                            multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
                            policy_chosen_logps, size_completion = get_batch_logps(
                                logits,
                                multimodal_labels,
                                is_encoder_decoder=False,
                                label_pad_token_id=-100,
                            ) 
                            # Calculate rejected_policy likelihood
                            output_rejected_policy: CausalLMOutputWithPast = vla(
                                input_ids=data_rejected["input_ids"].to(device_id),
                                attention_mask=data_rejected["attention_mask"].to(device_id),
                                pixel_values=data_rejected["pixel_values"].to(torch.bfloat16).to(device_id),
                                labels=data_rejected["labels"].to(device_id),
                            )
                            labels=data_rejected["labels"]
                            labels=labels.to(device_id)
                            logits=output_rejected_policy.logits
                            projected_patch_labels = torch.full(
                                (labels.shape[0], 256),
                                fill_value=-100,
                                dtype=labels.dtype,
                                device=device_id,
                            )
            
                            multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
                            policy_rejected_logps, size_completion = get_batch_logps(
                                logits,
                                multimodal_labels,
                                is_encoder_decoder=False,
                                label_pad_token_id=-100,
                            )

                            with torch.no_grad():
                                # Calculate chosen_reference likelihood
                                output_chosen_ref: CausalLMOutputWithPast = vla_ref(
                                    input_ids=data_chosen["input_ids"].to(device_id),
                                    attention_mask=data_chosen["attention_mask"].to(device_id),
                                    pixel_values=data_chosen["pixel_values"].to(torch.bfloat16).to(device_id),
                                    labels=data_chosen["labels"].to(device_id),
                                )
                                labels=data_chosen["labels"]
                                labels=labels.to(device_id)
                    
                                logits=output_chosen_ref.logits
                                projected_patch_labels = torch.full(
                                    (labels.shape[0], 256),
                                    fill_value=-100,
                                    dtype=labels.dtype,
                                    device=device_id,
                                )
                        
                                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
                                ref_chosen_logps, size_completion = get_batch_logps(
                                    logits,
                                    multimodal_labels,
                                    is_encoder_decoder=False,
                                    label_pad_token_id=-100,
                                )      
                                # Calculate rejected_reference likelihood
                                output_rejected_ref: CausalLMOutputWithPast = vla_ref(
                                    input_ids=data_rejected["input_ids"].to(device_id),
                                    attention_mask=data_rejected["attention_mask"].to(device_id),
                                    pixel_values=data_rejected["pixel_values"].to(torch.bfloat16).to(device_id),
                                    labels=data_rejected["labels"].to(device_id),
                                )
                                labels=data_rejected["labels"]
                                labels=labels.to(device_id)

                                logits=output_rejected_ref.logits
                                projected_patch_labels = torch.full(
                                    (labels.shape[0], 256),
                                    fill_value=-100,
                                    dtype=labels.dtype,
                                    device=device_id,
                                )

                                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
                                ref_rejected_logps, size_completion = get_batch_logps(
                                    logits,
                                    multimodal_labels,
                                    is_encoder_decoder=False,
                                    label_pad_token_id=-100,
                                )                                        
                            losses, chosen_rewards, rejected_rewards = dpo_loss(
                                policy_chosen_logps,
                                policy_rejected_logps,
                                ref_chosen_logps,
                                ref_rejected_logps,
                                device_id=device_id
                            )
                            #Calculate loss of this step 
                            loss_step=0.1*(policy_chosen_logps-policy_rejected_logps-ref_chosen_logps+ref_rejected_logps)
                            loss_sum+=loss_step
                            logps_chosen+=policy_chosen_logps
                            logps_rejected+=policy_rejected_logps
                            chosen_rewards_sum+=chosen_rewards
                            rejected_rewards_sum+=rejected_rewards
                           
                            
                # Normalize loss to account for gradient accumulation
                    loss_sum=-F.logsigmoid(loss_sum)
                    chosen_rewards=chosen_rewards_sum/8
                    rejected_rewards=rejected_rewards_sum/8
                    logps_chosen/=8
                    logps_rejected/=8
                    normalized_loss = loss_sum / cfg.grad_accumulation_steps
                    # Backward pass
                    normalized_loss.backward()

                    reward_accuracies = (chosen_rewards > rejected_rewards).float()
                    # # # Store recent train metrics
                    recent_losses.append(loss_sum.item())
                    reward_accuracies = (chosen_rewards > rejected_rewards).float()
                    recent_acc.append(reward_accuracies)
                    gradient_step_idx = traj_count // cfg.grad_accumulation_steps

                    smoothened_loss = sum(recent_losses) / len(recent_losses)
                    reward_accuracies=sum(recent_acc)/len(recent_acc)
                    # # Push Metrics to W&B (every 10 gradient steps)
                    reward_margins=chosen_rewards - rejected_rewards
                    if distributed_state.is_main_process and gradient_step_idx % 1==0:    
                        wandb.log(
                            {"train_loss": smoothened_loss,"chosen_rewards": chosen_rewards,"reject_rewards": rejected_rewards, "reward_acc":reward_accuracies,"reward_margins":reward_margins,"logps_rejected":logps_rejected,"logps_chosen":logps_chosen}, step=gradient_step_idx
                        )

                    # Optimizer Step
                    if (traj_count + 1) % cfg.grad_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        progress.update()

                    # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                    if gradient_step_idx > 0 and gradient_step_idx % 200 == 0:
                        if distributed_state.is_main_process:
                            print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                            directory_name="d1121_check"
                            
                            # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                            save_dir = f"{adapter_dir}/{directory_name}" if cfg.use_lora else run_dir
                            os.makedirs(save_dir, exist_ok=True)
                            # Save Processor & Weights
                            processor.save_pretrained(run_dir)
                            vla.module.save_pretrained(save_dir)

                        dist.barrier()


if __name__ == "__main__":
    finetune()
