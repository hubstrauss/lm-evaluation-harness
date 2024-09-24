from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM 
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from lm_eval.models.utils import stop_sequences_criteria
from lm_eval import utils
from datetime import timedelta

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from typing import Union, Optional, List, Tuple, Literal
import os
import torch
from tqdm import tqdm
from packaging import version

eval_logger = utils.eval_logger

@register_model("residual_model")
class ResidualModel(HFLM):
    """
    Not optimized / implemented for multi-gpu usage
    """

    def __init__(
        self,
        pretrained: str, # path to teacher
        residual: Union[str, os.PathLike], #path to residual model
        backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ):
        LM.__init__(self)
        # Starting with using some code from huggingface.py
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        revision = str(revision)  # cast to string if not already one
        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.pretrained = pretrained
        self.path_to_residual_model = os.path.join(residual, "model_checkpoint")
        self.path_to_residual_tokenizer = os.path.join(residual, "tokenizer")
        self.torch_dtype = dtype
        # self.batch_size = batch_size
        eval_logger.warning("Setting rank to 0 - only single gpu eval fro now")

        self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
        self.add_bos_token = add_bos_token

        self._get_student_model_config()
        self._create_tokenizer()
        self._instantiate_model(teacher = False)
        self._instantiate_model(teacher = True)
        self._move_models()

        self.vocab_size = self.tokenizer.vocab_size
        self.truncation = truncation
        self.logits_cache = logits_cache
        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if gpus > 1:
            eval_logger.error("Multi gpu not supported !")
            raise
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Assuming single-process call to evaluate() or custom distributed integration"
            )

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

    def _get_student_model_config(self):
        self.student_config = AutoConfig.from_pretrained(self.path_to_residual_model, trust_remote_code= True)

    def _create_tokenizer(self):
        self.tokenizer_kwargs = {
            "use_fast": True,
            "trust_remote_code": True,
            "model_max_length": self.student_config.max_position_embeddings,  # Setting length of tokenizer to length of model,
            "add_bos_token": self.add_bos_token,  # This seems to work only with Llama models ...
            "add_eos_token": False,  # This seems to work only with Llama models ...
        }

        # Tokenizer is from the student model (or if ssm, from the student models we want to compare it from. I think it can be from teacher model too)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_residual_tokenizer, **self.tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _move_models(self):
        self.model_frozen.to(self.device)
        self.model_student.to(self.device)

    def _instantiate_model(self, teacher = False):
        _name = "frozen" if teacher else "student"
        setattr(self, f"model_{_name}", self.AUTO_MODEL_CLASS.from_pretrained(
            self.pretrained if teacher else self.path_to_residual_model,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ))

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        Taken from huggingface model but adapated to residual model
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                final_logits = self.model_frozen(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits + self.model_student(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                final_logits = self.model_frozen(inps).logits + self.model_student(inps).logits
            
            return final_logits

    @property
    def max_length(self):
        if self._max_length:
            return self._max_length
        else:
            return self.student_config.max_position_embeddings
    
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        # For now haven;t implemented a generation method for the residuals as we need to add the logits before generation ...
        raise NotImplementedError

