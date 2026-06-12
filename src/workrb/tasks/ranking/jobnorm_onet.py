import json
from importlib import resources

from datasets import load_dataset

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType

TARGET_SPACE = []
with resources.open_text("workrb.data", "onet_30.json") as f:
    TARGET_SPACE = json.load(f)


@register_task()
class ONETJobNormRanking(RankingTask):
    @property
    def name(self) -> str:
        return "ONET Job Title Normalization"

    @property
    def description(self) -> str:
        return "Normalize job titles to canonical O*NET occupation groups"

    @property
    def task_group(self) -> RankingTaskGroup:
        return RankingTaskGroup.JOB_NORMALIZATION

    @property
    def supported_query_languages(self) -> list[Language]:
        return [Language.EN]

    @property
    def supported_target_languages(self) -> list[Language]:
        return [Language.EN]

    @property
    def label_type(self) -> LabelType:
        return LabelType.SINGLE_LABEL

    @property
    def query_input_type(self) -> ModelInputType:
        return ModelInputType.JOB_TITLE

    @property
    def target_input_type(self) -> ModelInputType:
        return ModelInputType.JOB_TITLE

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> RankingDataset:
        """Load job normalization data for a specific split and dataset.

        Args:
            dataset_id: Dataset identifier (language code for this task)
            split: Dataset split to load

        Returns
        -------
            RankingDataset object
        """
        query_texts, target_indices = [], []

        ds = load_dataset("workforcedeveloper/JobBERT-ONET-evaluation-dataset")
        split_map = {split.VAL: "valid", split.TEST: "test"}
        ds_split = ds[split_map[split]]
        df = ds_split.to_pandas()
        for index, row in df.iterrows():
            if row["onet_job_title"] in TARGET_SPACE:
                query_texts.append(row["vacancy_job_title"])
                target_index = TARGET_SPACE.index(row["onet_job_title"])
                target_indices.append([target_index])

        return RankingDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=TARGET_SPACE,
            dataset_id=dataset_id,
        )
