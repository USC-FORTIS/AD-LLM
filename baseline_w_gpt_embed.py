import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from openai import OpenAI
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from pyod.models.so_gaal_new import SO_GAAL
from pyod.models.vae import VAE
from utils import SynthDataset, TextDataset, evaluate
from config import DefaultConfig, PrivacyConfig

default_config = DefaultConfig()
privacy_config = PrivacyConfig()

baseline_map = {
    "autoencoder": AutoEncoder,
    "deepsvdd": DeepSVDD,
    "ecod": ECOD,
    "iforest": IForest,
    "lof": LOF,
    "lunar": LUNAR,
    "so_gaal": SO_GAAL,
    "vae": VAE
}
# parameter setting follows the default values in NLP-ADBench
params_map = {
    "autoencoder": {"batch_size": 4, "epoch_num": 30, "contamination": 0.1},
    "deepsvdd": {"batch_size": 4, "use_ae": False, 
                 "epochs": 5, "contamination": 0.1, 
                 "random_state": 10},
    "ecod": {},
    "iforest": {},
    "lof": {},
    "lunar": {},
    "so_gaal": {"epoch_num": 30, "contamination": 0.1, "verbose": 2},
    "vae": {"batch_size": 4, "epoch_num": 30, "contamination": 0.1, 
            "beta": 0.8, "capacity": 0.2}
}


def init_gpt():
    client = OpenAI(
        organization="org-hl3WFJbV0CMfCWTIVmS7JbiA",
        project="proj_5GhbbJZ4xGLQCxfrHyyJU299",
        api_key=privacy_config.gpt_api_key
    )
    return client


def generate_embeddings(gpt_client, dataloader):
    embeddings = []
    for text_batch in tqdm(dataloader):
        response = gpt_client.embeddings.create(
            model="text-embedding-3-large",
            input=text_batch,
        )
        embeddings.append([item.embedding for item in response.data])

    embeddings = np.vstack(embeddings)

    return embeddings


def main(baseline_name="lunar"):
    # available baselines: 
    # ["autoencoder", "deepsvdd", "ecod", "iforest", "lof", "lunar", "so_gaal", "vae"]
    if baseline_name not in baseline_map:
        raise ValueError(f"Invalid baseline name: {baseline_name}")
    print(f"Baseline: {baseline_name}")
    
    gpt_client = init_gpt()

    test_dataset = TextDataset(default_config.data_name, model_name="gpt")
    test_X = test_dataset.get_X()
    test_dataloader = DataLoader(test_X, batch_size=default_config.batch_size, 
                                shuffle=False, drop_last=False)
    test_gt = test_dataset.get_labels()
    test_gt = np.array(test_gt)

    use_desc = ""
    if default_config._use_desc:
        use_desc = "_use_desc"
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, 'data')
    data_name = default_config.data_name
    part_file_path = os.path.join(data_dir, data_name, f"{data_name}_gpt_part_embeddings.npy")
    part_and_synth_file_path = os.path.join(data_dir, data_name, f"{data_name}_gpt_part_and_synth_embeddings{use_desc}.npy")
    test_file_path = os.path.join(data_dir, data_name, f"{data_name}_test_embeddings.npy")

    if not os.path.exists(part_file_path):
        part_X = SynthDataset(default_config.data_name, mode=0, model_name="gpt")
        part_dataloader = DataLoader(part_X, batch_size=default_config.batch_size, 
                                    shuffle=True, drop_last=False)
        part_embeddings = generate_embeddings(gpt_client, part_dataloader)
        np.save(part_file_path, part_embeddings)
    else:
        part_embeddings = np.load(part_file_path)

    if not os.path.exists(part_and_synth_file_path):
        part_and_synth_X = SynthDataset(default_config.data_name, mode=1, model_name="gpt")
        part_and_synth_dataloader = DataLoader(part_and_synth_X, batch_size=default_config.batch_size, 
                                            shuffle=True, drop_last=False)
        part_and_synth_embeddings = generate_embeddings(gpt_client, part_and_synth_dataloader)
        np.save(part_and_synth_file_path, part_and_synth_embeddings)
    else:
        part_and_synth_embeddings = np.load(part_and_synth_file_path)

    if not os.path.exists(test_file_path):
        test_embeddings = generate_embeddings(gpt_client, test_dataloader)
        np.save(test_file_path, test_embeddings)
    else:
        test_embeddings = np.load(test_file_path)
        
    print(f"Part embeddings shape: {part_embeddings.shape}")
    print(f"Part and synth embeddings shape: {part_and_synth_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    
    if baseline_name == "deepsvdd":
        detector_part = DeepSVDD(n_features=part_embeddings.shape[1], 
                                 **params_map[baseline_name])
        detector_part_and_synth = DeepSVDD(n_features=part_and_synth_embeddings.shape[1], 
                                           **params_map[baseline_name])
    else:
        detector_part = baseline_map[baseline_name](**params_map[baseline_name])
        detector_part_and_synth = baseline_map[baseline_name](**params_map[baseline_name])

    detector_part.fit(part_embeddings)
    detector_part_and_synth.fit(part_and_synth_embeddings)

    test_score_part = detector_part.predict_proba(test_embeddings)[:, -1]
    test_score_part_and_synth = detector_part_and_synth.predict_proba(test_embeddings)[:, -1]

    print("without synthetic data:")
    evaluate(test_gt, test_score_part)
    print("********************************")
    print("Part and synth:")
    evaluate(test_gt, test_score_part_and_synth)


if __name__ == "__main__":
    main(baseline_name="lunar")
