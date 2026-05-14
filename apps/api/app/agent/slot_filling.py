from dataclasses import dataclass


@dataclass(frozen=True)
class SlotFillResult:
    value: str | None
    error: str | None = None
    ambiguous: bool = False


def fill_target_column(user_message: str, available_columns: list[str]) -> SlotFillResult:
    candidate = _clean_candidate(user_message)
    if not candidate:
        return SlotFillResult(
            value=None,
            error=f"Which target column should I use? Available columns: {_format_columns(available_columns)}",
        )

    exact_matches = [column for column in available_columns if column == candidate]
    if len(exact_matches) == 1:
        return SlotFillResult(value=exact_matches[0])
    if len(exact_matches) > 1:
        return SlotFillResult(value=None, ambiguous=True)

    lowered = candidate.casefold()
    case_insensitive_matches = [
        column for column in available_columns if column.casefold() == lowered
    ]
    if len(case_insensitive_matches) == 1:
        return SlotFillResult(value=case_insensitive_matches[0])
    if len(case_insensitive_matches) > 1:
        return SlotFillResult(value=None, ambiguous=True)

    return SlotFillResult(
        value=None,
        error=(
            f"I could not find '{candidate}' in the dataset. "
            f"Available columns are: {_format_columns(available_columns)}"
        ),
    )


def _clean_candidate(value: str) -> str:
    value = value.strip()
    prefixes = ("use ", "target ", "predict ", "column ")
    lowered = value.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return value[len(prefix) :].strip(" .,!?:;\"'")
    return value.strip(" .,!?:;\"'")


def _format_columns(columns: list[str]) -> str:
    return ", ".join(columns) if columns else "(none)"

