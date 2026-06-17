# pipeline/features/embedding.py

import torch
import torchaudio
import numpy as np

from transformers import ClapModel, ClapProcessor
from sklearn.metrics.pairwise import cosine_similarity


class CLAPEmbeddingExtractor:

    def __init__(self, device=None):

        self.device = (
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print("using device: ", self.device)
        
        self.processor = ClapProcessor.from_pretrained(
            "laion/clap-htsat-unfused"
        )

        self.model = ClapModel.from_pretrained(
            "laion/clap-htsat-unfused"
        )

        self.model.to(self.device)
        self.model.eval()

    def extract_embedding_from_audio(
        self,
        y,
        sr
    ):
        # ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=0)

        y = y.astype(np.float32)

        inputs = self.processor(
            audio=y,
            sampling_rate=sr,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.get_audio_features(
                **inputs
            )

            if isinstance(outputs, torch.Tensor):
                emb = outputs

            elif hasattr(outputs, "pooler_output"):
                emb = outputs.pooler_output

            else:
                raise ValueError(
                    f"Unexpected output type: {type(outputs)}"
                )

        emb = emb.cpu().numpy()[0]

        return emb


def cosine_sim(a, b):

    return float(
        cosine_similarity([a], [b])[0][0]
    )