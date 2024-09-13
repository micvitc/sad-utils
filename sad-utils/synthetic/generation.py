"""Simple Generation pipeline"""

import argparse
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import TextGeneration


def create_pipeline():
    with Pipeline(
        name="simple-text-generation-pipeline",
        description="A simple text generation pipeline",
    ) as pipeline:
        load_dataset = LoadDataFromHub(
            name="load_dataset",
            output_mappings={"prompt": "instruction"},
        )

        text_generation = TextGeneration(
            name="text_generation",
            llm=OpenAILLM(model="gpt-3.5-turbo"),
        )

        load_dataset >> text_generation
    return pipeline, load_dataset, text_generation


def main(repo_id, split, temperature, max_new_tokens, output_repo_id):
    pipeline, load_dataset, text_generation = create_pipeline()

    distiset = pipeline.run(
        parameters={
            load_dataset.name: {
                "repo_id": repo_id,
                "split": split,
            },
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                    }
                }
            },
        },
    )

    distiset.push_to_hub(repo_id=output_repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a simple text generation pipeline."
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        default="distilabel-internal-testing/instruction-dataset-mini",
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (e.g., train, test, validation)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature for text generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens for text generation",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default="distilabel-example",
        help="Repository ID to push the generated dataset to",
    )

    args = parser.parse_args()

    main(
        repo_id=args.repo_id,
        split=args.split,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        output_repo_id=args.output_repo_id,
    )
