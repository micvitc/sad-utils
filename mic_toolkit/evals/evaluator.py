from ..connect import Agent
from ..metrics import Metric
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


class Evaluator:
    def __init__(self, agent: Agent = None):
        self.agent = agent
        self.metric = []

    def add_metric(self, metric: Metric):
        self.metric.append(metric)

    def evaluate(self, data: pd.DataFrame):
        for metric in self.metric:
            m = metric()
            tqdm.write(f"Evaluating {m.name}")
            data[f"{m.name}"] = data["output"].progress_map(m.predict)
        return data
