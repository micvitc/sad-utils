from ollama import Client
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


class Generator:
    """Generator class for synthetic data generation."""

    def __init__(
        self,
        endpoint: str,
        model: str,
    ):
        """Initializes the LLM Client and model.

        Args:
            endpoint (str): Endpoint for the LLM API. For Ollama it is usually "http://localhost:11434".
            model (str): Name of the model to use for generation. Find it using 'ollama list'.
        """
        self.client = Client(endpoint)
        self.model = model

    def generate_text(
        self,
        data: pd.DataFrame,
        system_prompt: str = "You are a helpful AI assistant. Please provide a response to the following user query:",
        max_tokens: int = None,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            data (pd.DataFrame): Dataframe with a single column of text data.
            system_prompt (_type_, optional): optional System prompt. Defaults to "You are a helpful AI assistant. Please provide a response to the following user query:".
            max_tokens (int, optional): max output tokens. Defaults to None.

        Returns:
            pd.DataFrame: Output dataframe with generated text.
        """

        def generate_response(text):
            options = {}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            return self.client.chat(
                model=self.model,
                messages=[
                    {"role": "assistant", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                options=options,
            )["message"]["content"]

        data["output"] = data[data.columns[0]].progress_apply(generate_response)

        return data

    def create_system_prompt(self, labels: list[str], query: str = "") -> str:
        labels_str = ", ".join(labels)
        if query:
            return f"Classify the following text into one of the following categories: {labels_str} based on {query}. Just answer with the label. Absolutely no context is needed."
        else:
            return f"Classify the following text into one of the following categories: {labels_str}. Just answer with the label. Absolutely no context is needed."

    def generate_labels(
        self,
        labels: list[str],
        data: pd.DataFrame,
        query: str = "",
        max_tokens: int = None,
        max_tries: int = 5,
    ) -> pd.DataFrame:
        """_summary_

        Args:
            labels (list[str]): List of labels to classify the data into.
            data (pd.DataFrame): Dataframe with a single column of text data.
            query (str, optional): Classification query. Defaults to "".
            max_tokens (int, optional): max output tokens. Defaults to None.
            max_tries (int, optional): max tries to get the correct label. Defaults to 5.

        Returns:
            pd.DataFrame: _description_
        """
        system_prompt = self.create_system_prompt(labels, query)

        def classify_text(text):
            options = {}
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )["message"]["content"]
            tries = max_tries
            while response not in labels and tries > 0:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You did not respond with just the label please respond again with the label only. Without any context or explanation"
                            + system_prompt,
                        },
                        {"role": "user", "content": text},
                    ],
                    options=options,
                )["message"]["content"]
                tries -= 1
            return response

        data["label"] = data[data.columns[0]].progress_apply(classify_text)
        return data


if __name__ == "__main__":
    pass
