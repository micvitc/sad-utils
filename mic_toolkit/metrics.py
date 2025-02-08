from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


class Metric:
    def __init__(self, model, tokenizer, pipeline_name):
        total_steps = 2

        pbar = tqdm(total=total_steps, desc="Downloading Metric Model")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model
        )
        pbar.update(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer
        )
        pbar.update(1)
        self.pipeline = pipeline(
            pipeline_name, model=self.model, tokenizer=self.tokenizer
        )
        pbar.update(1)
        pbar.close()

    def predict(self, data):
        return self.pipeline(data)


class Toxicity(Metric):
    def __init__(self, model="s-nlp/roberta_toxicity_classifier"):
        self.name = "toxicity"

        total_steps = 2

        pbar = tqdm(total=total_steps, desc="Downloading Toxicity Model")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model,
        )
        pbar.update(1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model
        )
        pbar.update(1)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        pbar.update(1)
        pbar.close()

    def predict(self, data):
        res = self.pipeline(data, top_k=None)
        return res[0]["score"] if res[0]["label"] == "toxic" else res[1]["score"]
