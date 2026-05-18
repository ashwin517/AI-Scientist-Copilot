from typing import Any

from sqlalchemy.orm import Session

from app.reports.report_service import (
    generate_project_report,
    latest_project_report,
    list_project_reports,
    report_summary_payload,
    report_tool_payload,
)


REQUIRED_REPORT_SECTIONS = [
    "Project Overview",
    "Available Data",
    "Uploaded Documents",
    "Model Results",
    "Simulation Results",
    "Optimization Results",
    "Workflow Recommendations",
    "Limitations",
    "Recommended Next Steps",
]


def generate_project_report_tool(
    db: Session,
    project_id: int,
) -> dict[str, Any]:
    report = generate_project_report(db, project_id)
    return report_tool_payload(report)


def list_reports(
    db: Session,
    project_id: int,
    limit: int = 10,
) -> dict[str, Any]:
    reports = list_project_reports(db, project_id)[:limit]
    return {
        "reports": [report_summary_payload(report) for report in reports],
        "count": len(reports),
    }


def explain_latest_report(
    db: Session,
    project_id: int,
) -> dict[str, Any]:
    report = latest_project_report(db, project_id)
    if report is None:
        return {
            "report_available": False,
            "message": "No generated reports exist for this project yet.",
        }

    content = report.content_markdown
    sections = _section_names(content)
    source_summary = report_tool_payload(report).get("source_summary", {})
    return {
        "report_available": True,
        "report_id": report.id,
        "title": report.title,
        "report_type": report.report_type,
        "created_at": report.created_at.isoformat(),
        "sections": sections,
        "summary": _report_contents_summary(sections, source_summary),
        "recommended_next_steps": _recommended_next_steps(source_summary),
    }


def review_latest_report(
    db: Session,
    project_id: int,
) -> dict[str, Any]:
    report = latest_project_report(db, project_id)
    if report is None:
        return {
            "report_available": False,
            "message": "No generated reports exist for this project yet.",
            "strengths": [],
            "missing_sections": REQUIRED_REPORT_SECTIONS,
            "weak_sections": [],
            "suggested_edits": ["Generate a project report before asking for a review."],
            "limitations": [],
        }

    content = report.content_markdown
    sections = _section_names(content)
    section_text = _section_text_by_name(content)
    missing_sections = [
        section for section in REQUIRED_REPORT_SECTIONS if section not in sections
    ]
    weak_sections = _weak_sections(section_text)
    source_summary = report_tool_payload(report).get("source_summary", {})
    return {
        "report_available": True,
        "report_id": report.id,
        "title": report.title,
        "report_type": report.report_type,
        "created_at": report.created_at.isoformat(),
        "sections": sections,
        "strengths": _strengths(sections, section_text),
        "missing_sections": missing_sections,
        "weak_sections": weak_sections,
        "suggested_edits": _suggested_edits(missing_sections, weak_sections),
        "limitations": _review_limitations(content, source_summary),
    }


def _section_names(markdown: str) -> list[str]:
    return [
        line[3:].strip()
        for line in markdown.splitlines()
        if line.startswith("## ")
    ]


def _section_text_by_name(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for line in markdown.splitlines():
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
            continue
        if current_section is not None:
            sections[current_section].append(line)
    return {
        section: "\n".join(lines).strip()
        for section, lines in sections.items()
    }


def _weak_sections(section_text: dict[str, str]) -> list[dict[str, str]]:
    weak: list[dict[str, str]] = []
    for section in REQUIRED_REPORT_SECTIONS:
        text = section_text.get(section, "").strip()
        if not text:
            weak.append({"section": section, "reason": "Section is empty."})
            continue
        if _is_placeholder_text(text):
            weak.append(
                {
                    "section": section,
                    "reason": "Section mostly states that no saved evidence is available.",
                }
            )
            continue
        if section == "Model Results" and not _model_section_is_clear(text):
            weak.append(
                {
                    "section": section,
                    "reason": "Model explanation should include model type, target, metrics, and top features when available.",
                }
            )
        if section == "Limitations" and len(_bullet_lines(text)) < 2:
            weak.append(
                {
                    "section": section,
                    "reason": "Limitations section should include at least two grounded caveats.",
                }
            )
    return weak


def _strengths(
    sections: list[str],
    section_text: dict[str, str],
) -> list[str]:
    strengths: list[str] = []
    if all(section in sections for section in REQUIRED_REPORT_SECTIONS):
        strengths.append("Includes all expected report sections.")
    if section_text.get("Recommended Next Steps"):
        strengths.append("Provides recommended next steps.")
    model_text = section_text.get("Model Results", "")
    if _model_section_is_clear(model_text):
        strengths.append("Model section names the model, target, and metrics.")
    limitations_text = section_text.get("Limitations", "")
    if len(_bullet_lines(limitations_text)) >= 2:
        strengths.append("Limitations section includes multiple grounded caveats.")
    return strengths or ["Report has a saved markdown structure that can be reviewed."]


def _suggested_edits(
    missing_sections: list[str],
    weak_sections: list[dict[str, str]],
) -> list[str]:
    edits: list[str] = []
    for section in missing_sections:
        edits.append(f"Add a {section} section.")
    for weak_section in weak_sections:
        section = weak_section.get("section", "section")
        if section == "Model Results":
            edits.append(
                "Expand Model Results with target column, model type, key metrics, top features, and an interpretation caveat."
            )
        elif section == "Limitations":
            edits.append(
                "Strengthen Limitations with data coverage, model validity, and simulation/optimization caveats."
            )
        else:
            edits.append(f"Add more project-specific evidence to {section}.")
    return edits[:6] or [
        "No structural edits are required; consider adding domain interpretation and citations from uploaded documents."
    ]


def _review_limitations(content: str, source_summary: Any) -> list[str]:
    limitations = [
        "Review is based only on saved markdown content.",
        "The tool does not verify scientific correctness outside saved workspace state.",
    ]
    if "No documents are saved" in content:
        limitations.append("Document evidence is absent from this report.")
    if isinstance(source_summary, dict):
        if not source_summary.get("latest_model_run"):
            limitations.append("No saved model run was available when the report was generated.")
    return limitations


def _report_contents_summary(sections: list[str], source_summary: Any) -> str:
    project_name = None
    if isinstance(source_summary, dict):
        project = source_summary.get("project")
        if isinstance(project, dict):
            project_name = project.get("name")
    section_text = ", ".join(sections[:9]) if sections else "no detected sections"
    if project_name:
        return f"The report summarizes {project_name} across: {section_text}."
    return f"The report contains: {section_text}."


def _recommended_next_steps(source_summary: Any) -> list[str]:
    if not isinstance(source_summary, dict):
        return []
    steps = source_summary.get("recommended_next_steps", [])
    return [str(step) for step in steps] if isinstance(steps, list) else []


def _is_placeholder_text(text: str) -> bool:
    placeholder_markers = (
        "No datasets are saved",
        "No documents are saved",
        "No model runs are saved",
        "No simulation runs are saved",
        "No optimization runs are saved",
        "No workflow runs are saved",
        "No next steps are available",
    )
    stripped = text.strip()
    return any(marker in stripped for marker in placeholder_markers)


def _model_section_is_clear(text: str) -> bool:
    if "No model runs are saved" in text:
        return False
    required_terms = ("Model:", "Target column:", "Metrics:")
    return all(term in text for term in required_terms)


def _bullet_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip().startswith("- ")]
