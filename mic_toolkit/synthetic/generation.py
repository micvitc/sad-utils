from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration
from datasets import Dataset


class TextGenerationPipeline:
    """
    Simple Text Generation Pipeline.
    """

    def __init__(self, model_name: str, base_url: str, api_key: str):
        """Setup pipeline with LLM paramters.

        Args:
            model_name (str): Name of the model.
            base_url (str): URL endpoint for the model.
            api_key (str): API key.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.pipeline = self.create_pipeline()

    def create_pipeline(self) -> Pipeline:
        """Create the text generation pipeline.

        Returns:
            Pipeline: Text Generation Pipeline.
        """
        with Pipeline(
            name="simple-text-generation-pipeline",
            description="A simple text generation pipeline",
        ) as pipeline:
            TextGeneration(
                name="text_generation",
                llm=OpenAILLM(
                    model=self.model_name,
                    base_url=self.base_url,
                    api_key=self.api_key,
                ),
            )

        return pipeline

    def run_pipeline(
        self, dataset: Dataset, temperature: float = 0.7, max_new_tokens: int = 512
    ) -> Dataset:
        """
        Executes the text generation pipeline on the input dataset.

        Args:
            dataset: The input dataset to process.
            temperature: The temperature for text generation.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Dataset with generated text.
        """
        try:
            distiset = self.pipeline.run(
                dataset=dataset,
                parameters={
                    "text_generation": {
                        "llm": {
                            "generation_kwargs": {
                                "temperature": temperature,
                                "max_new_tokens": max_new_tokens,
                            }
                        }
                    },
                },
            )
            return distiset

        except Exception as e:
            raise e


if __name__ == "__main__":
    pass
