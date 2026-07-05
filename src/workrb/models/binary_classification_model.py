"""Binary classification model wrapper for WorkRB with proper label space tracking.

This module provides a binary classification model that classifies inputs into
one of two classes using Hugging Face's text-classification pipeline.
"""

import logging

import torch
from transformers import pipeline

from workrb.models.base import ModelInterface
from workrb.registry import register_model
from workrb.types import ModelInputType

logger = logging.getLogger(__name__)


@register_model()
class BinaryClassificationModel(ModelInterface):
    """
    Binary classification model for WorkRB tasks.

    This model wraps a sentence transformer encoder with a binary classification head.
    It tracks the exact label space (must be exactly 2 classes) and ensures predictions
    align correctly with task labels during evaluation.

    The model outputs logits for each class, which can be converted to probabilities
    using softmax or sigmoid depending on the use case.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        label_space: list[str] | None = None,
    ):
        """
        Initialize binary classification model with label space tracking.

        Args:
            model_name: Name of the pre-trained text classification model from Hugging Face.
                       Default: "distilbert-base-uncased-finetuned-sst-2-english"
                       (a model pre-trained for binary sentiment classification)
            label_space: Ordered list of exactly 2 label names (e.g., ["negative", "positive"]).
                        This MUST match the task's label space exactly.
                        If None, defaults to ["negative", "positive"].

        Raises
        ------
            ValueError: If label_space is not None and doesn't contain exactly 2 labels.
        """
        if label_space is None:
            label_space = ["negative", "positive"]

        if len(label_space) != 2:
            raise ValueError(
                f"Binary classification requires exactly 2 labels, "
                f"but got {len(label_space)}: {label_space}"
            )

        self._label_space = label_space
        self.model_name = model_name

        # Determine device
        self.device = 0 if torch.cuda.is_available() else -1

        # Load text classification pipeline
        self.pipeline = pipeline(
            task="text-classification",
            model=model_name,
            device=self.device,
            top_k=2,  # Get scores for both classes
        )

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return f"BinaryClassifier-{self.model_name.split('/')[-1]}"

    @property
    def description(self) -> str:
        return (
            f"Binary classification model using {self.model_name}. "
            f"Classes: {self._label_space[0]}, {self._label_space[1]}"
        )

    @property
    def classification_label_space(self) -> list[str] | None:
        """
        Ordered list of label names corresponding to classification outputs.

        Returns the exact label ordering used during training. This is critical
        for ensuring model outputs align with task labels during evaluation.

        Returns
        -------
            List of exactly 2 label names, e.g., ["negative", "positive"]
        """
        return self._label_space

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """
        Compute binary classification scores for texts.

        Args:
            texts: List of input texts to classify
            targets: List of target class labels (must be exactly ["negative", "positive"]
                    or the labels provided during initialization, in the same order)
            input_type: Type of input (e.g., JOB_TITLE, SKILL_NAME)
            target_input_type: Type of target (unused for this model, kept for interface compatibility)

        Returns
        -------
            Tensor of shape (n_texts, 2) where:
            - Column 0 corresponds to self._label_space[0]
            - Column 1 corresponds to self._label_space[1]
            Values are scores (not logits or probabilities, but model-output scores).

        Raises
        ------
            ValueError: If targets don't match the model's label space exactly.
        """
        # Validate that provided targets match model's internal label space
        if targets != self._label_space:
            if len(targets) != 2:
                raise ValueError(
                    f"Binary classification requires exactly 2 targets, "
                    f"but got {len(targets)}: {targets}"
                )
            if set(targets) != set(self._label_space):
                raise ValueError(
                    f"Target labels don't match model's label space. "
                    f"Model expects: {self._label_space} "
                    f"but got: {targets}"
                )
            # Same labels, different order
            logger.warning(
                "Target label order doesn't match model's internal label space. "
                f"Model expects: {self._label_space} but got: {targets}. "
                "This will cause incorrect predictions. Consider reordering or retraining."
            )

        # Classify texts using the pipeline
        results = self.pipeline(texts)

        # Convert pipeline results to tensor in the correct label order
        outputs_list = []
        for result_list, text in zip(results, texts):
            # result_list is a list of dicts with 'label' and 'score' keys
            scores = {}
            for item in result_list:
                label = item["label"]
                score = item["score"]
                scores[label] = score

            # Create output in the order of self._label_space
            output_row = []
            for label in self._label_space:
                # Try to match the label with pipeline output
                # Pipeline labels may be uppercase (e.g., 'POSITIVE', 'NEGATIVE')
                # or lowercase depending on the model
                score = self._get_score_for_label(label, scores)
                output_row.append(score)

            outputs_list.append(output_row)

        return torch.tensor(outputs_list, dtype=torch.float32)

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        """
        Compute ranking scores using binary classification scores.

        This model is designed for binary classification. For ranking tasks,
        it can only rank the 2 classes it was trained on.

        Args:
            queries: List of query texts
            targets: List of target texts (must be the 2 class labels)
            query_input_type: Type of query input
            target_input_type: Type of target input

        Returns
        -------
            Tensor of shape (n_queries, 2) with classification scores

        Raises
        ------
            ValueError: If targets don't match the model's 2 classes.
        """
        # Validate that targets match the model's label space
        if targets != self._label_space:
            if len(targets) != 2:
                raise ValueError(
                    "Cannot use binary classification model for ranking with non-binary targets. "
                    f"Model has 2 classes but task has {len(targets)} targets."
                )
            if set(targets) != set(self._label_space):
                raise ValueError(
                    "Cannot use binary classification model for ranking: target labels don't match "
                    f"model's label space. Model expects: {self._label_space} but got: {targets}"
                )
            # Same labels, different order
            raise ValueError(
                "Cannot use binary classification model for ranking: target label order doesn't match. "
                f"Model expects: {self._label_space} but got: {targets}"
            )

        # Compute classification scores and use for ranking
        # Since targets match label space, classification scores are valid ranking scores
        return self.compute_classification(
            texts=queries,
            targets=targets,
            input_type=query_input_type,
            target_input_type=target_input_type,
        )

    def get_class_index(self, label: str) -> int:
        """
        Get the output index for a specific class label.

        Args:
            label: Class label (must be in label_space)

        Returns
        -------
            Index of the label in the model's output (0 or 1)

        Raises
        ------
            ValueError: If label is not in label_space
        """
        if label not in self._label_space:
            raise ValueError(f"Label '{label}' not in model's label space: {self._label_space}")
        return self._label_space.index(label)

    def get_predictions_as_labels(self, logits: torch.Tensor) -> list[str]:
        """
        Convert logits to predicted class labels.

        Args:
            logits: Tensor of shape (n_samples, 2) with classification logits

        Returns
        -------
            List of predicted labels from label_space
        """
        predicted_indices = torch.argmax(logits, dim=1)
        return [self._label_space[idx] for idx in predicted_indices.tolist()]

    def get_predictions_as_probabilities(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Convert scores to class probabilities using softmax.

        Args:
            scores: Tensor of shape (n_samples, 2) with classification scores

        Returns
        -------
            Tensor of shape (n_samples, 2) with probabilities summing to 1
        """
        return torch.softmax(scores, dim=1)

    def _get_score_for_label(self, label: str, scores: dict) -> float:
        """
        Get the score for a label from the pipeline results, handling label format variations.

        The text-classification pipeline may return labels in different formats
        (e.g., 'POSITIVE' vs 'positive', 'LABEL_1' vs '1').

        Args:
            label: Target label from label_space
            scores: Dictionary mapping pipeline labels to scores

        Returns
        -------
            Score for the label, or 0.0 if label not found

        Raises
        ------
            ValueError: If the label cannot be found in any format
        """
        # Try exact match first
        if label in scores:
            return scores[label]

        # Try case-insensitive match
        label_upper = label.upper()
        for key, score in scores.items():
            if key.upper() == label_upper:
                return score

        # Try matching the first character or substring
        for key, score in scores.items():
            if label.lower() in key.lower() or key.lower() in label.lower():
                return score

        # If no match found, raise an error
        raise ValueError(
            f"Could not find label '{label}' in pipeline results. "
            f"Available labels: {list(scores.keys())}"
        )
