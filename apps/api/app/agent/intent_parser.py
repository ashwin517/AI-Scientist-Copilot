import re

from app.agent.tool_schemas import ToolIntent


class IntentParser:
    def parse(self, message: str) -> ToolIntent:
        normalized = _normalize(message)
        arguments: dict[str, object] = {}
        dataset_id = _extract_dataset_id(normalized)
        if dataset_id is not None:
            arguments["dataset_id"] = dataset_id
            arguments["latest"] = False

        if _contains_any(
            normalized,
            ("previous models", "model runs", "trained models", "model results"),
        ):
            return _intent("list_model_runs", arguments, 0.9)

        if _is_document_question(normalized):
            arguments["question"] = message.strip()
            arguments["top_k"] = 5
            return _intent("answer_document_question", arguments, 0.9)

        if _contains_any(
            normalized,
            ("train model", "train a model", "predict", "using "),
        ):
            target_column = _extract_target_column(normalized)
            if target_column:
                arguments["target_column"] = target_column
            return _intent("train_baseline_model", arguments, 0.86)

        if _contains_any(
            normalized,
            ("missing values", "null values", "incomplete columns"),
        ):
            return _intent("show_missing_values", arguments, 0.92)

        if _contains_any(
            normalized,
            (
                "latest dataset",
                "load dataset",
                "summarize dataset",
                "summarize my dataset",
                "dataset summary",
                "columns",
            ),
        ):
            arguments.setdefault("latest", True)
            return _intent("get_dataset_summary", arguments, 0.85)

        if _contains_any(
            normalized,
            ("list datasets", "what datasets", "show datasets"),
        ):
            return _intent("list_datasets", arguments, 0.93)

        return ToolIntent(
            requires_tool=False,
            confidence=0.0,
            explanation="No approved tool intent matched.",
        )


def _intent(
    tool_name: str,
    arguments: dict[str, object],
    confidence: float,
) -> ToolIntent:
    return ToolIntent(
        requires_tool=True,
        tool_name=tool_name,
        arguments=arguments,
        confidence=confidence,
        explanation=f"Rule-based match for {tool_name}.",
    )


def _normalize(message: str) -> str:
    return re.sub(r"\s+", " ", message.strip().lower())


def _contains_any(message: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in message for phrase in phrases)


def _is_document_question(message: str) -> bool:
    if _contains_any(
        message,
        (
            "what does the paper say",
            "according to the document",
            "according to the paper",
            "in the uploaded pdf",
            "uploaded pdf",
            "uploaded paper",
            "uploaded document",
            "based on the uploaded document",
            "based on the uploaded paper",
            "based on the uploaded pdf",
            "what does the document mention",
            "what does the document say",
            "what does this document say",
            "what does this paper say",
            "document mention about",
            "paper mention about",
            "summarize the sop",
            "summarise the sop",
            "summarize sop",
            "summarise sop",
        ),
    ):
        return True

    return (
        _contains_any(message, ("document", "paper", "pdf", "sop"))
        and _contains_any(
            message,
            ("what", "summarize", "summarise", "according", "mention", "based on"),
        )
    )


def _extract_dataset_id(message: str) -> int | None:
    match = re.search(r"\bdataset\s+(\d+)\b", message)
    if not match:
        return None
    return int(match.group(1))


def _extract_target_column(message: str) -> str | None:
    patterns = (
        r"\busing\s+(.+?)(?:\s+as\s+target\b|$)",
        r"\bpredict\s+(.+?)(?:$|\s+from\b|\s+with\b|\s+using\b)",
        r"\btarget\s+(.+?)(?:$|\s+column\b)",
        r"\bwith\s+(.+?)\s+as\s+target\b",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        target = _clean_target(match.group(1))
        if target:
            return target
    return None


def _clean_target(value: str) -> str:
    value = re.sub(r"\b(column|target|please|for me)\b", "", value)
    value = value.strip(" .,!?:;\"'")
    return re.sub(r"\s+", " ", value)
