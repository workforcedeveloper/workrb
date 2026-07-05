from datasets import load_dataset

from workrb.registry import register_task
from workrb.tasks.abstract.base import DatasetSplit, Language
from workrb.tasks.abstract.classification_base import (
    ClassificationDataset,
    ClassificationTaskGroup,
    MultiLabelClassificationTask,
)
from workrb.types import ModelInputType


@register_task()
class USAJobsTaskStatementClassification(MultiLabelClassificationTask):
    def __init__(self, task_type: str = "lenient", **kwargs):
        """
        Initialize Task Statement Classification task.

        Args:
            task_type: Whether the model should be evaluated on the lenient task or the strict task
            **kwargs: Arguments passed to parent ClassificationTask (languages, split, etc.)
        """
        self.task_type = task_type
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "USAJobs.gov Task Statement Classification Task"

    @property
    def description(self) -> str:
        return "Identify whether a phrase is or isn't a task statement."

    @property
    def task_group(self) -> ClassificationTaskGroup:
        return ClassificationTaskGroup.JOB2SKILL  # EDIT

    @property
    def supported_query_languages(self) -> list[Language]:
        """Queries (job titles) can be in English for val; all ESCO languages for test."""
        return [Language.EN]  # EDIT

    @property
    def supported_target_languages(self) -> list[Language]:
        """Target skills vocabulary for the configured ESCO version."""
        return [Language.EN]  # EDIT

    @property
    def input_type(self) -> ModelInputType:
        """Input is job titles."""
        return ModelInputType.JOB_TITLE  # EDIT

    def load_dataset(self, dataset_id: str, split: DatasetSplit) -> ClassificationDataset:
        """Load job normalization data for a specific split and dataset.

        Args:
            dataset_id: Dataset identifier (language code for this task)
            split: Dataset split to load

        Returns
        -------
            ClassificationDataset object
        """
        query_texts, target_indices = [], []
        target_space = [0, 1]
        ds = load_dataset("loyoladatamining/usajobs_validation")
        assert "sample" in ds

        df = ds["sample"].to_pandas()

        for index, row in df.iterrows():
            query_texts.append(row["text"])
            target_indices.append(
                [row["task_strict"]] if self.task_type == "strict" else [row["task_lenient"]]
            )

        return ClassificationDataset(
            query_texts=query_texts,
            target_indices=target_indices,
            target_space=target_space,
            dataset_id=dataset_id,
        )

    @property
    def citation(self) -> str:
        """TODO:"""
        return """
@article{meisenbacher2025extracting,
  title={Extracting O* NET Features from the NLx Corpus to Build Public Use Aggregate Labor Market Data},
  author={Meisenbacher, Stephen and Nestorov, Svetlozar and Norlander, Peter},
  year={2025}
}
"""
