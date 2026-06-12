import json
from importlib import resources

from datasets import load_dataset

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, LabelType, Language
from workrb.tasks.abstract.ranking_base import RankingDataset, RankingTask, RankingTaskGroup
from workrb.types import ModelInputType

TARGET_SPACE = {}
GLOBAL_TARGET_INDICES = []
GLOBAL_TARGETS = []

with resources.open_text("workrb.data", "2022_naics.json") as f:
    TARGET_SPACE = json.load(f)
    GLOBAL_TARGET_INDICES = [key for key in TARGET_SPACE.keys()]
    GLOBAL_TARGETS = [value for value in TARGET_SPACE.values()]


@register_task()
class DescriptionToIndustryRanking(RankingTask):
    def __init__(self, all_naics: bool = False, **kwargs):
        self.all_naics = all_naics
        super().__init__(**kwargs)

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

        ds = load_dataset("workforcedeveloper/DSBS-evaluation-dataset")
        split_map = {split.VAL: "validation", split.TEST: "test"}
        ds_split = ds[split_map[split]]
        df = ds_split.to_pandas()
        df = df.dropna()

        for index, row in df.iterrows():
            if self.all_naics:
                company_index_list = []
                naics_labels = row["all_naics"].split("|")
                for naic_code in naics_labels:
                    if naic_code in TARGET_SPACE:
                        target_index = GLOBAL_TARGET_INDICES.index(naic_code)
                        company_index_list.append(target_index)
                query_texts.append(row["description"])
                target_indices.append(company_index_list)
            elif row["primary_naics"] in TARGET_SPACE:
                query_texts.append(row["description"])
                target_index = GLOBAL_TARGET_INDICES.index(row["primary_naics"])
                target_indices.append([target_index])

        return RankingDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=GLOBAL_TARGETS,
            dataset_id=dataset_id,
        )
