import os
import random
import numpy as np
import torch
from transformers import set_seed


class PrivacyConfig:
    # set your OpenAI API key, organization, and project
    _api_key = "xxxxxxxxxxxxxx"
    _organization = "xxxxxxxxxxxxxx"
    _project = "xxxxxxxxxxxxxx"

    @property
    def gpt_api_key(self):
        return self._api_key
    
    @property
    def organization(self):
        return self._organization
    
    @property
    def project(self):
        return self._project


class DefaultConfig:
    # available data_name: ["agnews", "bbc", "movie_review", "N24News", "sms_spam"]
    data_name = "bbc"
    batch_size = 4

    # 1: "Normal Only" setting, 2: "Normal + Anomaly" setting.
    _ad_setting = 1

    # set True to use category description in anomaly detection
    # need to generate description before using it
    _use_desc = False

    # set the available cuda devices #
    _cuda_devices = "0, 1"

    # change the model id to use different models
    # note that in unsupervised model selection, we use "o1-preview"
    # if you wish to explore other models, you can change the model id in `select_gpt.py`
    _llama_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    _gpt_model_id = "gpt-4o"
    # _gpt_model_id = "o1-preview"

    _seed = 42
    _model_torch_type = torch.bfloat16

    _max_new_tokens = 512
    _more_max_new_tokens = 4096
    
    # number of keyword groups for each category in data augmentation -- synthetic data generation
    _num_keyword_groups_act = 50
    # add x more groups to account for unexpected behaviors (sometimes LLMs cannot follow the exact number of groups)
    _num_keyword_groups = _num_keyword_groups_act + 5

    # error symbol to handle the error in the anomaly detection
    # no need to change this value, and do not change to the values greater than 0
    _error_symbol = -1

    def __init__(self):
        # set available cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = self._cuda_devices
        
        # set seed for reproducibility
        random.seed(self._seed)
        np.random.seed(self._seed)
        set_seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"# of GPU: {torch.cuda.device_count()}")
        else:
            print("No GPU available")

        print(f"Dataset Name: {self.data_name}")
        print(f"AD Setting: {self._ad_setting}")
        print(f"Use Description: {self._use_desc}")


    @property
    def seed(self):
        return self._seed
    
    @property
    def device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # enable MPS in Apple M-series devices
        device = torch.device("mps" if torch.backends.mps.is_available() else device)
        return device
    
    @property
    def llama_model_id(self):
        return self._llama_model_id
    
    @property
    def gpt_model_id(self):
        return self._gpt_model_id
    
    @property
    def gpt_api_version(self):
        return self._gpt_api_version
    
    @property
    def model_torch_type(self):
        return self._model_torch_type
    
    @property
    def ad_setting(self):
        return self._ad_setting
    
    @property
    def use_desc(self):
        return self._use_desc
    
    @property
    def max_new_tokens(self):
        return self._max_new_tokens
    
    @property
    def more_max_new_tokens(self):
        return self._more_max_new_tokens
    
    # not recommended to use these two decorators together in the future
    @classmethod
    @property
    def error_symbol(self):
        return self._error_symbol
    
    @property
    def num_keyword_groups(self):
        return self._num_keyword_groups
    
    @property
    def num_keyword_groups_act(self):
        return self._num_keyword_groups_act
    