import re

from app.agent.tool_schemas import ToolIntent


class IntentParser:
    def parse(
        self,
        message: str,
        project_memory: dict[str, object] | None = None,
    ) -> ToolIntent:
        normalized = _normalize(message)
        arguments: dict[str, object] = {}
        dataset_id = _extract_dataset_id(normalized)
        if dataset_id is not None:
            arguments["dataset_id"] = dataset_id
            arguments["latest"] = False

        model_run_id = _extract_model_run_id(normalized)
        if model_run_id is not None:
            arguments["model_run_id"] = model_run_id

        target_column = _extract_set_target_column(normalized)
        if target_column:
            return _intent(
                "upsert_project_memory",
                {
                    "key": "selected_target_column",
                    "value": target_column,
                    "memory_type": "user_decision",
                    "source": "user_instruction",
                    "validate_target_column": True,
                },
                0.94,
            )

        if _is_memory_list_request(normalized):
            return _intent("list_project_memory", arguments, 0.93)

        remembered_fact = _extract_remembered_fact(message)
        if remembered_fact:
            return _intent(
                "upsert_project_memory",
                {
                    "key": _memory_key_for_fact(remembered_fact),
                    "value": remembered_fact,
                    "memory_type": "project_note",
                    "source": "user_instruction",
                },
                0.92,
            )

        forget_label = _extract_forget_label(normalized)
        if forget_label is not None:
            return _intent(
                "delete_project_memory",
                {"label": forget_label},
                0.92,
            )

        if _is_model_explanation(normalized):
            return _intent("explain_latest_model", arguments, 0.9)

        if _is_optimization_simulation_comparison_request(normalized):
            return _intent(
                "explain_latest_optimization",
                {"include_latest_simulation_comparison": True},
                0.94,
            )

        if _is_next_experiment_request(normalized):
            return _intent("recommend_next_experiment", {}, 0.94)

        if _is_workflow_comparison_request(normalized):
            return _intent("compare_workflow_runs", {}, 0.94)

        if _is_workflow_history_request(normalized):
            return _intent("list_workflow_runs", {}, 0.93)

        if _is_workflow_explanation_request(normalized):
            return _intent("explain_latest_workflow", {}, 0.93)

        if _is_report_list_request(normalized):
            return _intent("list_reports", {}, 0.93)

        if _is_report_review_request(normalized):
            return _intent("review_latest_report", {}, 0.94)

        if _is_report_explanation_request(normalized):
            return _intent("explain_latest_report", {}, 0.93)

        if _is_report_generation_request(normalized):
            return _intent("generate_project_report", {}, 0.94)

        if _is_project_analysis_request(normalized):
            return _intent("run_project_analysis_workflow", {}, 0.94)

        if _is_optimization_history_request(normalized):
            return _intent("list_optimization_runs", {}, 0.92)

        if _is_optimization_explanation_request(normalized):
            return _intent("explain_latest_optimization", {}, 0.93)

        if _is_simulation_history_request(normalized):
            return _intent("list_simulation_runs", {}, 0.92)

        if _is_simulation_explanation_request(normalized):
            return _intent("explain_latest_simulation", {}, 0.92)

        if _is_simulation_comparison_request(normalized):
            return _intent("compare_simulation_runs", {}, 0.92)

        if _is_batch_reactor_optimization_request(normalized):
            optimization_arguments = _extract_optimization_arguments(normalized)
            return _intent(
                "optimize_batch_reactor",
                optimization_arguments,
                0.93,
            )

        if _is_batch_reactor_simulation_request(normalized):
            simulation_arguments = _extract_simulation_arguments(normalized)
            return _intent(
                "run_batch_reactor_simulation",
                simulation_arguments,
                0.91,
            )

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


def _extract_model_run_id(message: str) -> int | None:
    match = re.search(r"\b(?:model\s+run|model)\s+#?(\d+)\b", message)
    if not match:
        return None
    return int(match.group(1))


def _is_model_explanation(message: str) -> bool:
    return _contains_any(
        message,
        (
            "explain the model",
            "explain latest model",
            "explain my model",
            "explain model",
            "explain latest model",
            "what does the model mean",
            "interpret the model",
            "summarize the model",
            "summarise the model",
            "important features",
            "feature importance",
            "why is performance poor",
            "why is the model performing",
            "why is the model performing this way",
            "what should i try next",
            "try next",
        ),
    )


def _is_memory_list_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "what do you remember",
            "what are you remembering",
            "show memory",
            "show project memory",
            "list memory",
            "list project memory",
            "what is in memory",
        ),
    )


def _is_batch_reactor_simulation_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "simulate batch reactor",
            "run reactor simulation",
            "reactor simulation",
            "simulate yield",
            "simulate reactor",
            "batch reactor simulation",
            "what if temperature",
        ),
    )


def _is_batch_reactor_optimization_request(message: str) -> bool:
    if _contains_any(
        message,
        (
            "optimize batch reactor",
            "optimize the batch reactor",
            "optimise batch reactor",
            "optimise the batch reactor",
            "maximize yield",
            "maximise yield",
            "limit impurity",
            "limiting impurity",
            "suggest operating conditions",
            "find best conditions",
            "find conditions that maximize yield",
            "find conditions that maximise yield",
            "suggest conditions for high yield",
            "high yield",
        ),
    ):
        return True

    return (
        _contains_any(message, ("optimize", "optimise", "maximize", "maximise", "best"))
        and _contains_any(message, ("batch reactor", "yield", "impurity", "conditions"))
    )


def _is_optimization_explanation_request(message: str) -> bool:
    if _contains_any(
        message,
        (
            "explain optimization",
            "explain the optimization",
            "explain latest optimization",
            "explain the latest optimization",
            "latest optimization",
            "why were these conditions selected",
            "why were the conditions selected",
            "why these conditions",
            "why selected",
            "optimization result",
            "optimization results",
        ),
    ):
        return True

    return (
        _contains_any(message, ("explain", "why"))
        and _contains_any(message, ("optimization", "optimisation", "conditions"))
    )


def _is_optimization_history_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "optimization history",
            "optimisation history",
            "show optimizations",
            "show optimisations",
            "list optimizations",
            "list optimisations",
            "previous optimizations",
            "previous optimisations",
            "saved optimizations",
            "saved optimisations",
        ),
    )


def _is_next_experiment_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "top candidates",
            "top candidate experiments",
            "show top candidate experiments",
            "next experiment",
            "recommend next experiment",
            "what experiment should i run next",
            "what should i run next",
            "what should we run next",
            "suggest next experiment",
        ),
    )


def _is_project_analysis_request(message: str) -> bool:
    if _contains_any(
        message,
        (
            "analyze this project",
            "analyse this project",
            "analyze the project",
            "analyse the project",
            "what should i do next",
            "give me project status",
            "summarize the project",
            "summarise the project",
            "recommend next steps",
            "suggest what i should do next",
            "project status",
            "status of this project",
        ),
    ):
        return True

    return (
        _contains_any(message, ("summarize", "summarise", "analyze", "analyse"))
        and _contains_any(message, ("project", "workspace"))
    )


def _is_workflow_history_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "workflow history",
            "project analysis history",
            "analysis history",
            "show workflow runs",
            "show workflows",
            "list workflow runs",
            "list workflows",
            "previous workflows",
            "previous analyses",
            "previous analysis runs",
        ),
    )


def _is_workflow_explanation_request(message: str) -> bool:
    if _contains_any(
        message,
        (
            "explain the latest project analysis",
            "explain latest project analysis",
            "explain the latest workflow",
            "explain latest workflow",
            "latest workflow",
            "latest project analysis",
            "last workflow",
            "last project analysis",
            "last analysis",
            "last recommendation",
            "last recommendations",
            "what did the last workflow recommend",
            "what did the latest workflow recommend",
            "what did the last analysis recommend",
            "what did the latest analysis recommend",
        ),
    ):
        return True

    return (
        _contains_any(message, ("recommend", "recommendation"))
        and _contains_any(message, ("last workflow", "latest workflow", "last analysis"))
    )


def _is_workflow_comparison_request(message: str) -> bool:
    if _contains_any(
        message,
        (
            "compare the last two analyses",
            "compare last two analyses",
            "compare analyses",
            "compare analysis runs",
            "compare workflow runs",
            "compare workflows",
            "compare the last two workflows",
            "compare last two workflows",
        ),
    ):
        return True

    return (
        _contains_any(message, ("compare", "comparison"))
        and _contains_any(message, ("workflow", "workflows", "analysis", "analyses"))
    )


def _is_report_generation_request(message: str) -> bool:
    if _contains_any(
        message,
        (
            "generate a project report",
            "generate project report",
            "generate report",
            "create project report",
            "create a project report",
            "create an analysis report",
            "analysis report",
            "summarize this project as a report",
            "summarise this project as a report",
            "prepare a technical summary",
            "technical summary",
            "write a project report",
            "prepare report",
        ),
    ):
        return True

    return (
        _contains_any(message, ("generate", "create", "prepare", "write"))
        and _contains_any(message, ("report", "technical summary"))
    )


def _is_report_list_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "list reports",
            "show reports",
            "report history",
            "show generated reports",
            "previous reports",
            "saved reports",
        ),
    )


def _is_report_review_request(message: str) -> bool:
    if _contains_any(
        message,
        (
            "review the latest report",
            "review latest report",
            "review report",
            "what is missing from the report",
            "what is missing in the report",
            "how can i improve this report",
            "improve this report",
            "improve report",
            "does the report clearly explain the model",
            "check the report limitations section",
            "check limitations",
            "report limitations",
            "limitations section",
        ),
    ):
        return True

    return (
        _contains_any(message, ("missing", "improve", "review", "check"))
        and _contains_any(message, ("report", "limitations"))
    )


def _is_report_explanation_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "explain latest report",
            "explain the latest report",
            "explain report",
            "latest report",
            "summarize latest report",
            "summarise latest report",
            "what does the report contain",
            "what is in the report",
        ),
    )


def _is_optimization_simulation_comparison_request(message: str) -> bool:
    return (
        _contains_any(message, ("compare", "comparison"))
        and _contains_any(message, ("optimization", "optimisation"))
        and _contains_any(message, ("simulation", "latest simulation"))
    )


def _is_simulation_history_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "simulation history",
            "show simulations",
            "show my simulations",
            "list simulations",
            "previous simulations",
            "saved simulations",
        ),
    )


def _is_simulation_explanation_request(message: str) -> bool:
    return _contains_any(
        message,
        (
            "explain latest simulation",
            "explain the latest simulation",
            "explain my latest simulation",
            "explain the simulation",
            "summarize latest simulation",
            "summarise latest simulation",
        ),
    )


def _is_simulation_comparison_request(message: str) -> bool:
    if _contains_any(
        message,
        (
            "compare simulations",
            "compare simulation runs",
            "compare last two simulations",
            "compare the last two simulations",
            "why did impurity increase",
            "why impurity increased",
            "what happens if i increase temperature",
            "what happens if temperature increases",
        ),
    ):
        return True

    return (
        _contains_any(message, ("compare", "why did", "what happens if"))
        and _contains_any(message, ("simulation", "impurity", "temperature"))
    )


def _extract_simulation_arguments(message: str) -> dict[str, object]:
    arguments: dict[str, object] = {}

    temperature = _extract_number_before_or_after(
        message,
        before_patterns=(r"\btemperature\s+(?:is\s+)?", r"\bat\s+"),
        after_patterns=(r"\s*(?:c|°c|degrees c|deg c)\b",),
    )
    if temperature is not None:
        arguments["temperature"] = temperature

    batch_time = _extract_number_before_or_after(
        message,
        before_patterns=(r"\bbatch[_\s-]*time\s+(?:is\s+)?", r"\bfor\s+"),
        after_patterns=(r"\s*(?:minutes|minute|min|mins)\b",),
    )
    if batch_time is not None:
        arguments["batch_time"] = batch_time

    initial_concentration = _extract_number_before_or_after(
        message,
        before_patterns=(
            r"\binitial[_\s-]*concentration\s+(?:is\s+)?",
            r"\binitial\s+",
        ),
        after_patterns=(r"\s*(?:m|mol/l|concentration)\b",),
    )
    if initial_concentration is not None:
        arguments["initial_concentration"] = initial_concentration

    catalyst_factor = _extract_number_before_or_after(
        message,
        before_patterns=(r"\bcatalyst[_\s-]*factor\s+(?:is\s+)?", r"\bcatalyst\s+"),
        after_patterns=(r"\s*x\s+catalyst\b",),
    )
    if catalyst_factor is not None:
        arguments["catalyst_factor"] = catalyst_factor

    return arguments


def _extract_optimization_arguments(message: str) -> dict[str, object]:
    arguments: dict[str, object] = {}

    impurity_limit = _extract_number_before_or_after(
        message,
        before_patterns=(
            r"\b(?:max(?:imum)?|limit|limiting)\s+impurity\s+(?:to\s+|at\s+)?",
            r"\bimpurity\s+(?:below|under|<=|less than)\s+",
        ),
        after_patterns=(r"\s*(?:max\s+)?impurity\b",),
    )
    if impurity_limit is not None:
        arguments["max_final_impurity"] = impurity_limit

    penalty_weight = _extract_number_before_or_after(
        message,
        before_patterns=(r"\bpenalty(?:[_\s-]*weight)?\s+(?:is\s+)?",),
        after_patterns=(r"\s*penalty(?:[_\s-]*weight)?\b",),
    )
    if penalty_weight is not None:
        arguments["penalty_weight"] = penalty_weight

    return arguments


def _extract_number_before_or_after(
    message: str,
    *,
    before_patterns: tuple[str, ...],
    after_patterns: tuple[str, ...],
) -> float | None:
    number = r"(-?\d+(?:\.\d+)?)"
    for pattern in before_patterns:
        match = re.search(pattern + number, message)
        if match:
            return float(match.group(1))
    for pattern in after_patterns:
        match = re.search(number + pattern, message)
        if match:
            return float(match.group(1))
    return None


def _extract_remembered_fact(message: str) -> str | None:
    match = re.search(
        r"^\s*remember\s+(?:that\s+)?(.+?)(?:[.!?]\s*)?$",
        message.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    fact = match.group(1).strip(" .,!?:;\"'")
    return fact or None


def _extract_forget_label(message: str) -> str | None:
    match = re.search(r"\bforget(?:\s+the|\s+my|\s+this|\s+that)?\s+(.+)$", message)
    if not match:
        return None
    label = match.group(1).strip(" .,!?:;\"'")
    return label


def _extract_set_target_column(message: str) -> str | None:
    patterns = (
        r"\buse\s+(.+?)\s+as\s+(?:the\s+)?target(?:\s+column)?(?:\s+from\s+now\s+on)?$",
        r"\bset\s+(?:the\s+)?target(?:\s+column)?\s+to\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if not match:
            continue
        target = _clean_target(match.group(1))
        if target:
            return target
    return None


def _memory_key_for_fact(fact: str) -> str:
    normalized = fact.casefold()
    if _contains_any(normalized, ("project is about", "project domain", "about ")):
        return "project_domain_note"
    return "project_note"


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
