import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import requests
from app.agent.agent_service import AgentService
from app.agent.intent_parser import IntentParser
from app.agent.pending_action import get_pending_action
from app.agent.planner import create_execution_plan
from app.agent.tools.document_tools import (
    DOCUMENT_NOT_PROCESSED_MESSAGE,
    EMBEDDINGS_MISSING_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    NO_RELEVANT_CHUNKS_MESSAGE,
    answer_document_question as answer_document_question_tool,
)
from app.agent.tools.model_tools import train_baseline_model
from app.agent.tools.optimization_tools import optimize_batch_reactor
from app.agent.tools.simulation_tools import (
    compare_simulation_runs,
    explain_latest_simulation,
    run_batch_reactor_simulation,
)
from app.agent.tool_executor import ToolExecutor
from app.agent.tool_registry import ToolRegistry
from app.agent.tool_schemas import ToolIntent
from app.ai.embedding_client import EmbeddingResult, OllamaEmbeddingClient
from app.ai.rag_service import (
    build_grounded_prompt,
    answer_document_question,
)
from app.ai.retrieval_service import (
    RetrievedChunk,
    cosine_similarity,
    retrieve_relevant_chunks,
)
from app.db.base import Base
from app.db.models import (
    Dataset,
    Document,
    DocumentChunk,
    ModelRun,
    Project,
    ProjectMemory,
    SimulationRun,
    OptimizationRun,
)
from app.db.session import get_db
from app.routes.datasets import router as datasets_router
from app.routes.documents import router as documents_router
from app.routes.memory import router as memory_router
from app.routes.models import router as models_router
from app.routes.optimization import router as optimization_router
from app.routes.projects import router as projects_router
from app.routes.simulation import router as simulation_router
from app.simulation.batch_reactor import simulate_batch_reactor
from app.simulation.schemas import BatchReactorSimulationInput
from app.optimization.batch_reactor_optimizer import optimize_batch_reactor_grid
from app.optimization.schemas import BatchReactorOptimizationInput
from app.services.memory_service import (
    get_memory,
    list_memory,
    update_project_summary,
    upsert_memory,
)
from app.services.document_chunks import process_document, split_text_into_chunks
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


class IntentParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = IntentParser()

    def test_detects_list_datasets(self) -> None:
        intent = self.parser.parse("list datasets in this project")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "list_datasets")

    def test_detects_show_missing_values(self) -> None:
        intent = self.parser.parse("show missing values")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "show_missing_values")

    def test_extracts_target_column_for_training(self) -> None:
        intent = self.parser.parse("train a model using yield")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "train_baseline_model")
        self.assertEqual(intent.arguments["target_column"], "yield")

    def test_detects_document_question(self) -> None:
        intent = self.parser.parse("what does the uploaded paper say about yield?")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "answer_document_question")
        self.assertEqual(
            intent.arguments["question"],
            "what does the uploaded paper say about yield?",
        )

    def test_detects_based_on_uploaded_document_question(self) -> None:
        intent = self.parser.parse("based on the uploaded document, what is yield?")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "answer_document_question")

    def test_detects_list_project_memory(self) -> None:
        intent = self.parser.parse("what do you remember about this project?")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "list_project_memory")

    def test_detects_remember_project_fact(self) -> None:
        intent = self.parser.parse(
            "remember that this project is about batch reactor yield optimization"
        )

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "upsert_project_memory")
        self.assertEqual(intent.arguments["key"], "project_domain_note")
        self.assertEqual(
            intent.arguments["value"],
            "this project is about batch reactor yield optimization",
        )

    def test_detects_forget_project_memory(self) -> None:
        intent = self.parser.parse("forget the target column")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "delete_project_memory")
        self.assertEqual(intent.arguments["label"], "target column")

    def test_detects_set_selected_target_column(self) -> None:
        intent = self.parser.parse("use yield_pct as the target column from now on")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "upsert_project_memory")
        self.assertEqual(intent.arguments["key"], "selected_target_column")
        self.assertEqual(intent.arguments["value"], "yield_pct")
        self.assertTrue(intent.arguments["validate_target_column"])

    def test_detects_explain_latest_model(self) -> None:
        intent = self.parser.parse("what are the important features?")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "explain_latest_model")

    def test_detects_batch_reactor_simulation(self) -> None:
        intent = self.parser.parse("what if temperature is 85C for 120 minutes")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "run_batch_reactor_simulation")
        self.assertEqual(intent.arguments["temperature"], 85.0)
        self.assertEqual(intent.arguments["batch_time"], 120.0)

    def test_detects_simulation_history(self) -> None:
        intent = self.parser.parse("show my simulation history")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "list_simulation_runs")

    def test_detects_explain_latest_simulation(self) -> None:
        intent = self.parser.parse("explain the latest simulation")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "explain_latest_simulation")

    def test_detects_compare_last_two_simulations(self) -> None:
        intent = self.parser.parse("compare the last two simulations")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "compare_simulation_runs")

    def test_detects_impurity_increase_question(self) -> None:
        intent = self.parser.parse("why did impurity increase?")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "compare_simulation_runs")

    def test_detects_batch_reactor_optimization(self) -> None:
        intent = self.parser.parse("maximize yield while limiting impurity")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "optimize_batch_reactor")

    def test_detects_latest_optimization_explanation(self) -> None:
        intent = self.parser.parse("explain the latest optimization")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "explain_latest_optimization")

    def test_detects_next_experiment_recommendation(self) -> None:
        intent = self.parser.parse("what experiment should I run next?")

        self.assertTrue(intent.requires_tool)
        self.assertEqual(intent.tool_name, "recommend_next_experiment")


class ToolExecutorTests(unittest.TestCase):
    def test_rejects_unknown_tools(self) -> None:
        executor = ToolExecutor()
        intent = ToolIntent(
            requires_tool=True,
            tool_name="drop_database",
            arguments={},
            confidence=0.99,
        )

        result = executor.execute(db=None, project_id=1, intent=intent)  # type: ignore[arg-type]

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Unknown or unapproved tool requested.")


class ToolRegistryTests(unittest.TestCase):
    def test_contains_expected_tools(self) -> None:
        registry = ToolRegistry()

        self.assertEqual(
            set(registry.names()),
            {
                "list_datasets",
                "get_dataset_summary",
                "show_missing_values",
                "train_baseline_model",
                "list_model_runs",
                "explain_latest_model",
                "answer_document_question",
                "list_project_memory",
                "upsert_project_memory",
                "delete_project_memory",
                "run_batch_reactor_simulation",
                "list_simulation_runs",
                "explain_latest_simulation",
                "compare_simulation_runs",
                "optimize_batch_reactor",
                "explain_latest_optimization",
                "list_optimization_runs",
                "recommend_next_experiment",
                "run_project_analysis_workflow",
                "list_workflow_runs",
                "explain_latest_workflow",
                "compare_workflow_runs",
                "generate_project_report",
                "list_reports",
                "explain_latest_report",
                "review_latest_report",
            },
        )


class PlannerTests(unittest.TestCase):
    def test_create_execution_plan_for_project_analysis_uses_registry_tools(self) -> None:
        registry = ToolRegistry()

        plan = create_execution_plan(
            "Analyze my project and suggest next experiments",
            {"latest_optimization_run_id": 1},
            registry=registry,
        )

        self.assertGreaterEqual(len(plan), 3)
        self.assertLessEqual(len(plan), 8)
        self.assertEqual(plan[0].tool_name, "list_project_memory")
        self.assertIn("recommend_next_experiment", [step.tool_name for step in plan])
        self.assertTrue(all(registry.get(step.tool_name) for step in plan))
        self.assertTrue(all(step.status == "pending" for step in plan))

    def test_create_execution_plan_ignores_single_tool_requests(self) -> None:
        plan = create_execution_plan(
            "summarize my dataset",
            {},
            registry=ToolRegistry(),
        )

        self.assertEqual(plan, [])


class ProjectMemoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Memory project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_memory_upsert_updates_existing_key(self) -> None:
        first = upsert_memory(
            self.db,
            self.project.id,
            "latest_dataset_id",
            1,
            memory_type="dataset",
            source="dataset_upload",
        )
        second = upsert_memory(
            self.db,
            self.project.id,
            "latest_dataset_id",
            2,
            memory_type="dataset",
            source="dataset_upload",
        )

        self.assertEqual(first.id, second.id)
        self.assertEqual(self.db.query(ProjectMemory).count(), 1)
        latest_dataset = get_memory(self.db, self.project.id, "latest_dataset_id")
        self.assertEqual(latest_dataset.value_json, "2")

    def test_memory_retrieval_is_project_scoped(self) -> None:
        other_project = Project(name="Other memory project")
        self.db.add(other_project)
        self.db.commit()
        self.db.refresh(other_project)
        upsert_memory(self.db, self.project.id, "selected_target_column", "yield")
        upsert_memory(self.db, other_project.id, "selected_target_column", "quality")

        memories = list_memory(self.db, self.project.id)

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].project_id, self.project.id)
        self.assertEqual(memories[0].value_json, '"yield"')

    def test_empty_project_summary_uses_empty_message(self) -> None:
        summary = update_project_summary(self.db, self.project.id)

        self.assertEqual(summary, "No project summary has been created yet.")
        memory = get_memory(self.db, self.project.id, "project_summary")
        self.assertIsNotNone(memory)
        self.assertEqual(memory.value_json, '"No project summary has been created yet."')


class ProjectMemoryRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Memory route project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)

        app = FastAPI()
        app.include_router(memory_router)

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_list_and_delete_project_memory(self) -> None:
        upsert_memory(
            self.db,
            self.project.id,
            "project_domain_note",
            "batch reactor yield optimization",
            memory_type="project_note",
            source="test",
        )

        list_response = self.client.get(f"/projects/{self.project.id}/memory")
        delete_response = self.client.delete(
            f"/projects/{self.project.id}/memory/project_domain_note"
        )

        self.assertEqual(list_response.status_code, 200)
        memory_by_key = {item["key"]: item for item in list_response.json()}
        self.assertEqual(
            memory_by_key["project_domain_note"]["value"],
            "batch reactor yield optimization",
        )
        self.assertEqual(delete_response.status_code, 204)
        self.assertIsNone(get_memory(self.db, self.project.id, "project_domain_note"))


class BatchReactorSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Simulation project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)

        app = FastAPI()
        app.include_router(simulation_router)

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_ode_simulation_returns_valid_profiles(self) -> None:
        result = simulate_batch_reactor(
            BatchReactorSimulationInput(
                temperature=85,
                batch_time=120,
                initial_concentration=1.0,
                catalyst_factor=1.0,
            )
        )

        self.assertEqual(len(result.time_grid), len(result.CA_profile))
        self.assertEqual(len(result.time_grid), len(result.CB_profile))
        self.assertEqual(len(result.time_grid), len(result.CC_profile))
        self.assertGreater(len(result.time_grid), 2)
        self.assertGreaterEqual(result.final_yield, 0.0)
        self.assertGreaterEqual(result.final_impurity, 0.0)
        self.assertGreaterEqual(result.conversion, 0.0)

    def test_simulation_run_is_persisted(self) -> None:
        response = self.client.post(
            f"/projects/{self.project.id}/simulation/batch-reactor",
            json={
                "temperature": 85,
                "batch_time": 120,
                "initial_concentration": 1.0,
                "catalyst_factor": 1.0,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["simulation_type"], "batch_reactor")
        self.assertGreaterEqual(body["final_yield"], 0.0)
        self.assertEqual(self.db.query(SimulationRun).count(), 1)
        simulation_memory = get_memory(
            self.db,
            self.project.id,
            "latest_simulation_run_id",
        )
        simulation_type_memory = get_memory(
            self.db,
            self.project.id,
            "latest_simulation_type",
        )
        self.assertIsNotNone(simulation_memory)
        self.assertIsNotNone(simulation_type_memory)
        self.assertEqual(simulation_memory.value_json, str(body["simulation_run_id"]))
        self.assertEqual(simulation_type_memory.value_json, '"batch_reactor"')

    def test_latest_simulation_can_be_explained(self) -> None:
        run_batch_reactor_simulation(
            self.db,
            self.project.id,
            temperature=85,
            batch_time=120,
        )

        explanation = explain_latest_simulation(self.db, self.project.id)

        self.assertTrue(explanation["simulation_available"])
        self.assertEqual(explanation["simulation_type"], "batch_reactor")
        self.assertIn("interpretation", explanation)
        self.assertIn("not calibrated for real chemistry", explanation["model_note"])

    def test_last_two_simulations_can_be_compared(self) -> None:
        run_batch_reactor_simulation(
            self.db,
            self.project.id,
            temperature=80,
            batch_time=120,
        )
        run_batch_reactor_simulation(
            self.db,
            self.project.id,
            temperature=90,
            batch_time=120,
        )

        comparison = compare_simulation_runs(self.db, self.project.id)

        self.assertTrue(comparison["comparison_available"])
        self.assertEqual(comparison["candidate"]["input"]["temperature"], 90)
        self.assertEqual(comparison["baseline"]["input"]["temperature"], 80)
        self.assertIn("temperature", comparison["input_differences"])
        self.assertIn("final_yield", comparison["result_differences"])

    def test_no_simulation_gives_clean_message(self) -> None:
        explanation = explain_latest_simulation(self.db, self.project.id)
        comparison = compare_simulation_runs(self.db, self.project.id)

        self.assertFalse(explanation["simulation_available"])
        self.assertIn("No simulation runs are saved", explanation["message"])
        self.assertFalse(comparison["comparison_available"])
        self.assertIn("No simulation runs are saved", comparison["message"])


class BatchReactorOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Optimization project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)

        app = FastAPI()
        app.include_router(optimization_router)

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_grid_optimizer_returns_best_candidate(self) -> None:
        result = optimize_batch_reactor_grid(_small_optimization_input())

        self.assertGreater(result.evaluated_candidates, 0)
        self.assertGreaterEqual(result.best_final_yield, 0.0)
        self.assertLessEqual(result.best_final_impurity, 1.0)
        self.assertIn("temperature_c", result.best_inputs)
        self.assertGreaterEqual(len(result.top_candidates), 1)

    def test_optimization_run_is_persisted(self) -> None:
        response = self.client.post(
            f"/projects/{self.project.id}/optimization/batch-reactor",
            json=_small_optimization_input().model_dump(),
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["optimization_type"], "batch_reactor")
        self.assertIn("best_inputs", body)
        self.assertGreaterEqual(body["best_final_yield"], 0.0)
        self.assertEqual(self.db.query(OptimizationRun).count(), 1)
        optimization_memory = get_memory(
            self.db,
            self.project.id,
            "latest_optimization_run_id",
        )
        optimization_type_memory = get_memory(
            self.db,
            self.project.id,
            "latest_optimization_type",
        )
        self.assertIsNotNone(optimization_memory)
        self.assertIsNotNone(optimization_type_memory)
        self.assertEqual(optimization_memory.value_json, str(body["optimization_run_id"]))
        self.assertEqual(optimization_type_memory.value_json, '"batch_reactor"')

    def test_agent_routes_optimization_request(self) -> None:
        result = optimize_batch_reactor(self.db, self.project.id)

        self.assertIn("best_inputs", result)
        self.assertEqual(result["optimization_type"], "batch_reactor")
        self.assertEqual(self.db.query(OptimizationRun).count(), 1)


def _small_optimization_input() -> BatchReactorOptimizationInput:
    return BatchReactorOptimizationInput(
        search_space={
            "temperature_c": {"min": 70.0, "max": 80.0, "steps": 2},
            "batch_time_min": {"min": 30.0, "max": 60.0, "steps": 2},
            "initial_concentration": {"min": 0.5, "max": 1.0, "steps": 2},
            "catalyst_factor": {"min": 0.5, "max": 1.0, "steps": 2},
        },
        top_k=3,
    )


class ProjectDeleteIdResetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)

        app = FastAPI()
        app.include_router(projects_router)

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_project_id_resets_after_all_projects_are_deleted(self) -> None:
        first = self.client.post("/projects", json={"name": "First"})
        self.assertEqual(first.status_code, 201)
        self.assertEqual(first.json()["id"], 1)

        delete_response = self.client.delete("/projects/1")
        self.assertEqual(delete_response.status_code, 204)

        second = self.client.post("/projects", json={"name": "Second"})
        self.assertEqual(second.status_code, 201)
        self.assertEqual(second.json()["id"], 1)

    def test_project_id_does_not_reset_while_projects_remain(self) -> None:
        first = self.client.post("/projects", json={"name": "First"})
        second = self.client.post("/projects", json={"name": "Second"})
        self.assertEqual(first.json()["id"], 1)
        self.assertEqual(second.json()["id"], 2)

        delete_response = self.client.delete("/projects/1")
        self.assertEqual(delete_response.status_code, 204)

        third = self.client.post("/projects", json={"name": "Third"})
        self.assertEqual(third.status_code, 201)
        self.assertEqual(third.json()["id"], 3)


class TrainBaselineModelToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Test project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_no_datasets_in_project(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "This project has no saved datasets yet. Please upload a CSV dataset first.",
        ):
            train_baseline_model(
                self.db,
                project_id=self.project.id,
                target_column="yield",
            )

    def test_dataset_exists_but_no_target_column_provided(self) -> None:
        self._create_dataset(
            [
                {"temperature": 20, "yield": 40},
                {"temperature": 21, "yield": 42},
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Please specify a target column. Available columns are: temperature, yield",
        ):
            train_baseline_model(self.db, project_id=self.project.id)

    def test_dataset_exists_but_target_column_is_invalid(self) -> None:
        self._create_dataset(
            [
                {"temperature": 20, "yield": 40},
                {"temperature": 21, "yield": 42},
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "Target column 'quality' was not found in this dataset. "
            "Available columns are: temperature, yield",
        ):
            train_baseline_model(
                self.db,
                project_id=self.project.id,
                target_column="quality",
            )

    def test_dataset_exists_and_valid_target_column_trains(self) -> None:
        dataset = self._create_dataset(
            [
                {"temperature": 20, "pressure": 1.0, "yield": 40},
                {"temperature": 21, "pressure": 1.1, "yield": 42},
                {"temperature": 22, "pressure": 1.2, "yield": 44},
                {"temperature": 23, "pressure": 1.3, "yield": 46},
                {"temperature": 24, "pressure": 1.4, "yield": 48},
                {"temperature": 25, "pressure": 1.5, "yield": 50},
            ]
        )

        result = train_baseline_model(
            self.db,
            project_id=self.project.id,
            target_column="yield",
        )

        self.assertEqual(result["target_column"], "yield")
        self.assertEqual(result["dataset_id"], dataset.id)
        self.assertIsNotNone(result["model_run_id"])
        self.assertEqual(result["model_result"]["problem_type"], "regression")
        self.assertEqual(self.db.query(ModelRun).count(), 1)

    def _create_dataset(self, rows: list[dict[str, object]]) -> Dataset:
        dataset = Dataset(
            project_id=self.project.id,
            filename="test.csv",
            row_count=len(rows),
            column_count=len(rows[0]) if rows else 0,
            raw_data_json=json.dumps(rows),
        )
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        return dataset


class DatasetUploadRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Upload project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)

        app = FastAPI()
        app.include_router(datasets_router)

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_project_csv_upload_persists_dataset(self) -> None:
        response = self.client.post(
            f"/projects/{self.project.id}/datasets/upload",
            files={
                "file": (
                    "test.csv",
                    b"temperature,yield\n20,40\n21,42\n",
                    "text/csv",
                )
            },
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["dataset"]["filename"], "test.csv")
        self.assertEqual(body["preview"]["rows"], 2)

        datasets = self.db.query(Dataset).filter_by(project_id=self.project.id).all()
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0].filename, "test.csv")
        memory = get_memory(self.db, self.project.id, "latest_dataset_id")
        self.assertIsNotNone(memory)
        self.assertEqual(memory.value_json, str(datasets[0].id))
        filename_memory = get_memory(
            self.db,
            self.project.id,
            "latest_dataset_filename",
        )
        count_memory = get_memory(self.db, self.project.id, "dataset_count")
        self.assertIsNotNone(filename_memory)
        self.assertIsNotNone(count_memory)
        self.assertEqual(filename_memory.value_json, '"test.csv"')
        self.assertEqual(count_memory.value_json, "1")
        summary_memory = get_memory(self.db, self.project.id, "project_summary")
        self.assertIsNotNone(summary_memory)
        self.assertIn("test.csv", summary_memory.value_json)


class DocumentRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Document project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)
        self.storage_dir = tempfile.TemporaryDirectory()
        self.settings_patch = patch(
            "app.services.documents.settings",
            SimpleNamespace(storage_dir=self.storage_dir.name),
        )
        self.settings_patch.start()
        self.embedding_patch = patch(
            "app.services.document_chunks.OllamaEmbeddingClient",
            return_value=FakeEmbeddingClient(),
        )
        self.embedding_patch.start()

        app = FastAPI()
        app.include_router(documents_router)

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.embedding_patch.stop()
        self.settings_patch.stop()
        self.storage_dir.cleanup()
        self.db.close()
        self.engine.dispose()

    def test_project_document_upload_persists_document(self) -> None:
        response = self.client.post(
            f"/projects/{self.project.id}/documents/upload",
            files={
                "file": (
                    "notes.txt",
                    b"experiment notes",
                    "text/plain",
                )
            },
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["document"]["filename"], "notes.txt")
        self.assertEqual(body["document"]["mime_type"], "text/plain")
        self.assertIsNotNone(body["document"]["extracted_text_path"])
        self.assertEqual(body["processing"]["chunk_count"], 1)
        self.assertEqual(body["processing"]["embedded_chunk_count"], 1)

        documents = self.db.query(Document).filter_by(project_id=self.project.id).all()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].filename, "notes.txt")
        memory = get_memory(self.db, self.project.id, "latest_document_id")
        self.assertIsNotNone(memory)
        self.assertEqual(memory.value_json, str(documents[0].id))
        filename_memory = get_memory(
            self.db,
            self.project.id,
            "latest_document_filename",
        )
        count_memory = get_memory(self.db, self.project.id, "document_count")
        self.assertIsNotNone(filename_memory)
        self.assertIsNotNone(count_memory)
        self.assertEqual(filename_memory.value_json, '"notes.txt"')
        self.assertEqual(count_memory.value_json, "1")
        summary_memory = get_memory(self.db, self.project.id, "project_summary")
        self.assertIsNotNone(summary_memory)
        self.assertIn("notes.txt", summary_memory.value_json)
        chunks = self.db.query(DocumentChunk).filter_by(document_id=documents[0].id).all()
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].project_id, self.project.id)
        self.assertIsNotNone(chunks[0].embedding_json)

    def test_project_document_debug_reports_processing_state(self) -> None:
        upload = self.client.post(
            f"/projects/{self.project.id}/documents/upload",
            files={"file": ("notes.txt", b"experiment notes", "text/plain")},
        )
        document_id = upload.json()["document"]["id"]

        response = self.client.get(f"/projects/{self.project.id}/documents/debug")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["documents_count"], 1)
        self.assertEqual(body["chunks_count"], 1)
        self.assertEqual(body["embedded_chunks_count"], 1)
        self.assertEqual(body["documents"][0]["id"], document_id)
        self.assertTrue(body["documents"][0]["has_extracted_text"])
        self.assertEqual(body["documents"][0]["chunk_count"], 1)
        self.assertEqual(body["documents"][0]["embedded_chunk_count"], 1)

    def test_project_documents_list_and_detail_are_project_scoped(self) -> None:
        upload = self.client.post(
            f"/projects/{self.project.id}/documents/upload",
            files={"file": ("notes.md", b"# Notes", "text/markdown")},
        )
        document_id = upload.json()["document"]["id"]

        list_response = self.client.get(f"/projects/{self.project.id}/documents")
        detail_response = self.client.get(
            f"/projects/{self.project.id}/documents/{document_id}"
        )

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()), 1)
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.json()["id"], document_id)

    def test_project_document_process_endpoint_rebuilds_chunks(self) -> None:
        upload = self.client.post(
            f"/projects/{self.project.id}/documents/upload",
            files={
                "file": (
                    "long-notes.txt",
                    ("alpha " * 450).encode("utf-8"),
                    "text/plain",
                )
            },
        )
        document_id = upload.json()["document"]["id"]

        response = self.client.post(
            f"/projects/{self.project.id}/documents/{document_id}/process"
        )

        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json()["chunk_count"], 1)
        chunks = (
            self.db.query(DocumentChunk)
            .filter_by(document_id=document_id, project_id=self.project.id)
            .all()
        )
        self.assertEqual(len(chunks), response.json()["chunk_count"])

    def test_project_document_delete_removes_document_and_chunks(self) -> None:
        upload = self.client.post(
            f"/projects/{self.project.id}/documents/upload",
            files={"file": ("notes.txt", b"experiment notes", "text/plain")},
        )
        document_id = upload.json()["document"]["id"]
        self.assertEqual(
            self.db.query(DocumentChunk).filter_by(document_id=document_id).count(),
            1,
        )

        response = self.client.delete(
            f"/projects/{self.project.id}/documents/{document_id}"
        )

        self.assertEqual(response.status_code, 204)
        self.assertIsNone(self.db.get(Document, document_id))
        self.assertEqual(
            self.db.query(DocumentChunk).filter_by(document_id=document_id).count(),
            0,
        )


class FakeEmbeddingClient:
    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(embedding=[0.1, 0.2, 0.3])


class FailingEmbeddingClient:
    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(embedding=None, error="Ollama unavailable")


class DocumentChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Chunking project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)
        self.storage_dir = tempfile.TemporaryDirectory()
        self.text_path = f"{self.storage_dir.name}/notes.txt"
        with open(self.text_path, "w", encoding="utf-8") as file:
            file.write("alpha " * 450)
        self.document = Document(
            project_id=self.project.id,
            filename="notes.txt",
            file_path=self.text_path,
            mime_type="text/plain",
            file_size=100,
            extracted_text_path=self.text_path,
        )
        self.db.add(self.document)
        self.db.commit()
        self.db.refresh(self.document)

    def tearDown(self) -> None:
        self.storage_dir.cleanup()
        self.db.close()
        self.engine.dispose()

    def test_chunking_creates_multiple_chunks(self) -> None:
        chunks = split_text_into_chunks(
            "alpha " * 450,
            chunk_size=1000,
            chunk_overlap=150,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))

    def test_processing_links_chunks_to_document_and_project(self) -> None:
        result = process_document(
            self.db,
            self.document,
            embedding_client=FakeEmbeddingClient(),
        )

        self.assertGreater(result.chunk_count, 1)
        self.assertEqual(result.chunk_count, result.embedded_chunk_count)
        chunks = self.db.query(DocumentChunk).filter_by(document_id=self.document.id).all()
        self.assertEqual(len(chunks), result.chunk_count)
        self.assertTrue(all(chunk.project_id == self.project.id for chunk in chunks))

    def test_processing_keeps_chunks_when_embedding_fails(self) -> None:
        result = process_document(
            self.db,
            self.document,
            embedding_client=FailingEmbeddingClient(),
        )

        self.assertGreater(result.chunk_count, 1)
        self.assertEqual(result.embedded_chunk_count, 0)
        self.assertEqual(result.embedding_error_count, result.chunk_count)
        chunks = self.db.query(DocumentChunk).filter_by(document_id=self.document.id).all()
        self.assertTrue(all(chunk.embedding_json is None for chunk in chunks))


class EmbeddingClientTests(unittest.TestCase):
    def test_embedding_client_handles_ollama_failure_gracefully(self) -> None:
        client = OllamaEmbeddingClient(url="http://localhost:11434/api/embeddings")

        with patch(
            "app.ai.embedding_client.requests.post",
            side_effect=requests.RequestException("offline"),
        ):
            result = client.embed_text("hello")

        self.assertIsNone(result.embedding)
        self.assertIn("offline", result.error)


class RetrievalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Retrieval project")
        self.other_project = Project(name="Other retrieval project")
        self.db.add_all([self.project, self.other_project])
        self.db.commit()
        self.db.refresh(self.project)
        self.db.refresh(self.other_project)

        self.document = self._create_document(self.project.id, "paper.txt")
        self.other_document = self._create_document(
            self.other_project.id,
            "other-paper.txt",
        )
        self._create_chunk(
            self.document,
            chunk_index=0,
            content="reaction yield improves with temperature",
            embedding=[1.0, 0.0],
        )
        self._create_chunk(
            self.document,
            chunk_index=1,
            content="pressure changes were less relevant",
            embedding=[0.2, 0.8],
        )
        self._create_chunk(
            self.other_document,
            chunk_index=0,
            content="other project yield chunk",
            embedding=[1.0, 0.0],
        )

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_cosine_similarity_works(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_retrieval_returns_ranked_chunks(self) -> None:
        results = retrieve_relevant_chunks(
            self.db,
            project_id=self.project.id,
            query="yield",
            top_k=2,
            embedding_client=StaticEmbeddingClient([1.0, 0.0]),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk_index, 0)
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(results[0].filename, "paper.txt")

    def test_project_scoping_is_respected(self) -> None:
        results = retrieve_relevant_chunks(
            self.db,
            project_id=self.project.id,
            query="yield",
            top_k=5,
            embedding_client=StaticEmbeddingClient([1.0, 0.0]),
        )

        self.assertTrue(results)
        self.assertTrue(
            all(result.document_id == self.document.id for result in results)
        )

    def test_empty_project_returns_clean_response(self) -> None:
        empty_project = Project(name="Empty retrieval project")
        self.db.add(empty_project)
        self.db.commit()
        self.db.refresh(empty_project)

        results = retrieve_relevant_chunks(
            self.db,
            project_id=empty_project.id,
            query="yield",
            top_k=5,
            embedding_client=StaticEmbeddingClient([1.0, 0.0]),
        )

        self.assertEqual(results, [])

    def _create_document(self, project_id: int, filename: str) -> Document:
        document = Document(
            project_id=project_id,
            filename=filename,
            file_path=filename,
            mime_type="text/plain",
            file_size=10,
            extracted_text_path=filename,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def _create_chunk(
        self,
        document: Document,
        chunk_index: int,
        content: str,
        embedding: list[float],
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document.id,
            project_id=document.project_id,
            chunk_index=chunk_index,
            content=content,
            char_count=len(content),
            embedding_json=json.dumps(embedding),
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        return chunk


class StaticEmbeddingClient:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding

    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(embedding=self.embedding)


class RagServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_no_relevant_chunks_returns_clean_answer(self) -> None:
        with patch(
            "app.ai.rag_service.retrieve_relevant_chunks",
            return_value=[],
        ):
            answer = answer_document_question(
                self.db,
                project_id=1,
                question="What does this say about yield?",
            )

        self.assertIn("could not find relevant document chunks", answer.answer)
        self.assertEqual(answer.sources, [])

    def test_sources_are_returned(self) -> None:
        retrieved_chunk = self._retrieved_chunk()

        with patch(
            "app.ai.rag_service.retrieve_relevant_chunks",
            return_value=[retrieved_chunk],
        ), patch("app.ai.rag_service.requests.post") as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            mock_post.return_value.json.return_value = {
                "response": (
                    "The document says yield improved with temperature "
                    "[doc:paper.txt chunk:3]."
                )
            }

            answer = answer_document_question(
                self.db,
                project_id=1,
                question="What does this say about yield?",
            )

        self.assertIn("[doc:paper.txt chunk:3]", answer.answer)
        self.assertEqual(len(answer.sources), 1)
        self.assertEqual(answer.sources[0].document_id, 1)
        self.assertEqual(answer.sources[0].chunk_id, 12)

    def test_prompt_includes_retrieved_chunks(self) -> None:
        retrieved_chunk = self._retrieved_chunk()
        prompt = build_grounded_prompt(
            "What does this say about yield?",
            [retrieved_chunk],
        )

        self.assertIn("Answer only from the provided chunks.", prompt)
        self.assertIn("[doc:paper.txt chunk:3]", prompt)
        self.assertIn("Yield increased when reaction temperature rose.", prompt)

    def _retrieved_chunk(self) -> RetrievedChunk:
        return RetrievedChunk(
            document_id=1,
            filename="paper.txt",
            chunk_id=12,
            chunk_index=3,
            score=0.82,
            content="Yield increased when reaction temperature rose.",
            content_preview="Yield increased when reaction temperature rose.",
        )


class DocumentQuestionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Document tool project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)
        self.document = Document(
            project_id=self.project.id,
            filename="paper.txt",
            file_path="paper.txt",
            mime_type="text/plain",
            file_size=10,
            extracted_text_path="paper.txt",
        )
        self.db.add(self.document)
        self.db.commit()
        self.db.refresh(self.document)
        self.chunk = DocumentChunk(
            document_id=self.document.id,
            project_id=self.project.id,
            chunk_index=0,
            content="Relevant yield chunk.",
            char_count=len("Relevant yield chunk."),
            embedding_json=json.dumps([1.0, 0.0]),
        )
        self.db.add(self.chunk)
        self.db.commit()
        self.db.refresh(self.chunk)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_rag_tool_calls_rag_service(self) -> None:
        rag_answer = SimpleNamespace(
            answer="Grounded answer [doc:paper.txt chunk:1].",
            sources=[
                SimpleNamespace(
                    document_id=self.document.id,
                    filename="paper.txt",
                    chunk_id=4,
                    chunk_index=1,
                    score=0.91,
                    content_preview="Relevant chunk.",
                )
            ],
        )

        with patch(
            "app.agent.tools.document_tools.generate_document_answer",
            return_value=rag_answer,
        ) as mock_generate:
            result = answer_document_question_tool(
                self.db,
                project_id=self.project.id,
                question="What does the paper say?",
                top_k=3,
            )

        mock_generate.assert_called_once_with(
            self.db,
            project_id=self.project.id,
            question="What does the paper say?",
            top_k=3,
        )
        self.assertEqual(result["answer"], rag_answer.answer)
        self.assertEqual(result["sources"][0]["chunk_id"], 4)

    def test_rag_tool_reports_unprocessed_documents(self) -> None:
        self.db.delete(self.chunk)
        self.db.commit()

        result = answer_document_question_tool(
            self.db,
            project_id=self.project.id,
            question="What does the paper say?",
        )

        self.assertEqual(result["answer"], DOCUMENT_NOT_PROCESSED_MESSAGE)
        self.assertEqual(result["sources"], [])

    def test_rag_tool_reports_missing_embeddings(self) -> None:
        self.chunk.embedding_json = None
        self.db.add(self.chunk)
        self.db.commit()

        result = answer_document_question_tool(
            self.db,
            project_id=self.project.id,
            question="What does the paper say?",
        )

        self.assertEqual(result["answer"], EMBEDDINGS_MISSING_MESSAGE)
        self.assertEqual(result["sources"], [])

    def test_rag_tool_reports_zero_retrieval_results(self) -> None:
        rag_answer = SimpleNamespace(answer="No chunks.", sources=[])

        with patch(
            "app.agent.tools.document_tools.generate_document_answer",
            return_value=rag_answer,
        ):
            result = answer_document_question_tool(
                self.db,
                project_id=self.project.id,
                question="What does the paper say?",
            )

        self.assertEqual(result["answer"], NO_RELEVANT_CHUNKS_MESSAGE)
        self.assertEqual(result["sources"], [])


class ModelTrainingRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Model project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)
        self.dataset = Dataset(
            project_id=self.project.id,
            filename="training.csv",
            row_count=6,
            column_count=3,
            raw_data_json=json.dumps(
                [
                    {"temperature": 20, "pressure": 1.0, "yield": 40},
                    {"temperature": 21, "pressure": 1.1, "yield": 42},
                    {"temperature": 22, "pressure": 1.2, "yield": 44},
                    {"temperature": 23, "pressure": 1.3, "yield": 46},
                    {"temperature": 24, "pressure": 1.4, "yield": 48},
                    {"temperature": 25, "pressure": 1.5, "yield": 50},
                ]
            ),
        )
        self.db.add(self.dataset)
        self.db.commit()
        self.db.refresh(self.dataset)

        app = FastAPI()
        app.include_router(models_router)

        def override_get_db():
            db = self.session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_persistent_training_endpoint_creates_model_run(self) -> None:
        response = self.client.post(
            "/models/train",
            json={
                "project_id": self.project.id,
                "dataset_id": self.dataset.id,
                "target_column": "yield",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNotNone(body["model_run_id"])
        self.assertEqual(body["dataset_id"], self.dataset.id)
        self.assertEqual(self.db.query(ModelRun).count(), 1)
        model_memory = get_memory(self.db, self.project.id, "latest_model_run_id")
        target_memory = get_memory(self.db, self.project.id, "selected_target_column")
        task_memory = get_memory(self.db, self.project.id, "latest_task_type")
        self.assertIsNotNone(model_memory)
        self.assertIsNotNone(target_memory)
        self.assertIsNotNone(task_memory)
        self.assertEqual(model_memory.value_json, str(body["model_run_id"]))
        self.assertEqual(target_memory.value_json, '"yield"')
        self.assertEqual(task_memory.value_json, '"regression"')
        summary_memory = get_memory(self.db, self.project.id, "project_summary")
        self.assertIsNotNone(summary_memory)
        self.assertIn(f"#{body['model_run_id']}", summary_memory.value_json)
        self.assertIn("RandomForestRegressor regression", summary_memory.value_json)

    def test_transient_training_endpoint_still_works(self) -> None:
        response = self.client.post(
            "/models/train",
            json={
                "data": json.loads(self.dataset.raw_data_json),
                "target_column": "yield",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body["model_run_id"])
        self.assertEqual(body["problem_type"], "regression")


class PendingActionAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.db: Session = self.session_factory()
        self.project = Project(name="Pending project")
        self.db.add(self.project)
        self.db.commit()
        self.db.refresh(self.project)
        self.dataset = Dataset(
            project_id=self.project.id,
            filename="pending.csv",
            row_count=6,
            column_count=3,
            raw_data_json=json.dumps(
                [
                    {"temperature": 20, "pressure": 1.0, "yield": 40},
                    {"temperature": 21, "pressure": 1.1, "yield": 42},
                    {"temperature": 22, "pressure": 1.2, "yield": 44},
                    {"temperature": 23, "pressure": 1.3, "yield": 46},
                    {"temperature": 24, "pressure": 1.4, "yield": 48},
                    {"temperature": 25, "pressure": 1.5, "yield": 50},
                ]
            ),
        )
        self.db.add(self.dataset)
        self.db.commit()
        self.db.refresh(self.dataset)
        self.agent = AgentService()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_train_model_creates_pending_action_when_target_missing(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "train a model",
        )

        self.assertIn("Which target column should I use?", response.reply)
        self.assertEqual(response.message, response.reply)
        self.assertFalse(response.tool_used)
        self.assertEqual(response.pending_action["tool_name"], "train_baseline_model")
        self.assertEqual(response.pending_action["missing_fields"], ["target_column"])
        self.assertIn("yield", response.reply)
        pending_action = get_pending_action(self.db, self.project.id)
        self.assertIsNotNone(pending_action)
        self.assertEqual(pending_action.tool_name, "train_baseline_model")

    def test_multi_step_project_analysis_executes_plan_trace(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "Analyze my project and suggest next experiments",
        )

        self.assertTrue(response.tool_used)
        self.assertTrue(response.plan_executed)
        self.assertEqual(response.tool_name, "execution_plan")
        self.assertIsNotNone(response.steps_summary)
        self.assertGreaterEqual(len(response.steps_summary), 3)
        self.assertLessEqual(len(response.steps_summary), 8)
        self.assertIn("list_project_memory", response.tools_used)
        self.assertIn("get_dataset_summary", response.tools_used)
        self.assertIn("I did not train models", response.reply)
        self.assertEqual(self.db.query(SimulationRun).count(), 0)
        self.assertEqual(self.db.query(OptimizationRun).count(), 0)

    def test_list_memory_intent_returns_project_memory(self) -> None:
        upsert_memory(
            self.db,
            self.project.id,
            "project_domain_note",
            "batch reactor yield optimization",
            memory_type="project_note",
            source="test",
        )

        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "what do you remember about this project?",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "list_project_memory")
        self.assertIn("project_domain_note", response.reply)
        self.assertIn("batch reactor yield optimization", response.reply)

    def test_list_memory_formats_non_model_context(self) -> None:
        document = Document(
            project_id=self.project.id,
            filename="test_paper_batch_yield.pdf",
            file_path="test_paper_batch_yield.pdf",
            mime_type="application/pdf",
            file_size=100,
            extracted_text_path=None,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)

        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "what do you remember about this project?",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "list_project_memory")
        self.assertIn("I currently remember:", response.reply)
        self.assertIn("Latest document: test_paper_batch_yield.pdf", response.reply)
        self.assertIn("Documents uploaded: 1", response.reply)
        self.assertIn("Latest dataset: pending.csv", response.reply)
        self.assertIn("No trained models yet", response.reply)

    def test_list_memory_syncs_existing_document_from_previous_session(self) -> None:
        document = Document(
            project_id=self.project.id,
            filename="test_paper_batch_yield.pdf",
            file_path="test_paper_batch_yield.pdf",
            mime_type="application/pdf",
            file_size=100,
            extracted_text_path=None,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        self.assertIsNone(get_memory(self.db, self.project.id, "latest_document_id"))

        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "what do you remember about this project?",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "list_project_memory")
        self.assertIn("Latest document: test_paper_batch_yield.pdf", response.reply)
        self.assertIn("Documents uploaded: 1", response.reply)
        synced_memory = get_memory(self.db, self.project.id, "latest_document_filename")
        self.assertIsNotNone(synced_memory)
        self.assertEqual(synced_memory.value_json, '"test_paper_batch_yield.pdf"')

    def test_remember_fact_creates_project_memory(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "remember that this project is about batch reactor yield optimization",
        )

        memory = get_memory(self.db, self.project.id, "project_domain_note")
        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "upsert_project_memory")
        self.assertIsNotNone(memory)
        self.assertEqual(
            memory.value_json,
            '"this project is about batch reactor yield optimization"',
        )
        self.assertIn("I will remember", response.reply)
        summary_memory = get_memory(self.db, self.project.id, "project_summary")
        self.assertIsNotNone(summary_memory)
        self.assertIn(
            "this project is about batch reactor yield optimization",
            summary_memory.value_json,
        )

    def test_forget_target_deletes_selected_target_column(self) -> None:
        upsert_memory(
            self.db,
            self.project.id,
            "selected_target_column",
            "yield",
            memory_type="user_decision",
            source="test",
        )

        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "forget the target column",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "delete_project_memory")
        self.assertIsNone(get_memory(self.db, self.project.id, "selected_target_column"))
        self.assertIn("I forgot selected_target_column", response.reply)

    def test_forget_ambiguous_memory_asks_clarification(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "forget that",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "delete_project_memory")
        self.assertIn("Which project memory should I forget?", response.reply)

    def test_set_target_column_validates_and_updates_memory(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "use yield as the target column from now on",
        )

        memory = get_memory(self.db, self.project.id, "selected_target_column")
        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "upsert_project_memory")
        self.assertIsNotNone(memory)
        self.assertEqual(memory.value_json, '"yield"')
        self.assertIn("I will use yield as the target column from now on", response.reply)

    def test_agent_routes_simulation_request(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "simulate yield at 90C for 120 minutes",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "run_batch_reactor_simulation")
        self.assertEqual(response.tool_result["simulation_type"], "batch_reactor")
        self.assertEqual(response.tool_result["input"]["temperature"], 90.0)
        self.assertEqual(response.tool_result["input"]["batch_time"], 120.0)
        self.assertGreaterEqual(response.tool_result["final_yield"], 0.0)
        self.assertEqual(self.db.query(SimulationRun).count(), 1)

    def test_set_target_column_finds_column_in_non_active_dataset(self) -> None:
        yield_dataset = Dataset(
            project_id=self.project.id,
            filename="yield_dataset.csv",
            row_count=6,
            column_count=3,
            raw_data_json=json.dumps(
                [
                    {"run_id": 1, "temperature": 20, "yield_pct": 80},
                    {"run_id": 2, "temperature": 21, "yield_pct": 82},
                    {"run_id": 3, "temperature": 22, "yield_pct": 83},
                    {"run_id": 4, "temperature": 23, "yield_pct": 84},
                    {"run_id": 5, "temperature": 24, "yield_pct": 85},
                    {"run_id": 6, "temperature": 25, "yield_pct": 86},
                ]
            ),
        )
        self.db.add(yield_dataset)
        self.db.commit()
        self.db.refresh(yield_dataset)
        upsert_memory(
            self.db,
            self.project.id,
            "latest_dataset_id",
            self.dataset.id,
            memory_type="dataset",
            source="test",
        )

        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "use yield_pct as the target column from now on",
        )

        target_memory = get_memory(
            self.db,
            self.project.id,
            "selected_target_column",
        )
        dataset_memory = get_memory(self.db, self.project.id, "latest_dataset_id")
        filename_memory = get_memory(
            self.db,
            self.project.id,
            "latest_dataset_filename",
        )
        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "upsert_project_memory")
        self.assertEqual(target_memory.value_json, '"yield_pct"')
        self.assertEqual(dataset_memory.value_json, str(yield_dataset.id))
        self.assertEqual(filename_memory.value_json, '"yield_dataset.csv"')
        self.assertIn("using dataset yield_dataset.csv", response.reply)

    def test_invalid_target_column_returns_available_columns(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "use quality as the target column from now on",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "upsert_project_memory")
        self.assertIsNone(get_memory(self.db, self.project.id, "selected_target_column"))
        self.assertIn("Target column 'quality' was not found in any saved project dataset", response.reply)
        self.assertIn("Available columns are: temperature, pressure, yield.", response.reply)

    def test_train_model_uses_remembered_target_column(self) -> None:
        upsert_memory(
            self.db,
            self.project.id,
            "selected_target_column",
            "yield",
            memory_type="user_decision",
            source="test",
        )

        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "train a model",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "train_baseline_model")
        self.assertEqual(response.tool_result["target_column"], "yield")
        self.assertIn(
            "I used your remembered target column: yield.",
            response.reply,
        )
        self.assertIsNone(get_pending_action(self.db, self.project.id))

    def test_missing_target_memory_falls_back_to_clarification(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "train a model",
        )

        self.assertFalse(response.tool_used)
        self.assertIn("Which target column should I use?", response.reply)
        self.assertIsNotNone(get_pending_action(self.db, self.project.id))

    def test_memory_does_not_override_explicit_target_column(self) -> None:
        upsert_memory(
            self.db,
            self.project.id,
            "selected_target_column",
            "yield",
            memory_type="user_decision",
            source="test",
        )

        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "train a model using pressure",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "train_baseline_model")
        self.assertEqual(response.tool_result["target_column"], "pressure")
        self.assertNotIn("remembered target column", response.reply)

    def test_dataset_summary_response_includes_structured_tool_result(self) -> None:
        with patch(
            "app.agent.agent_service.requests.post",
            side_effect=requests.RequestException,
        ):
            response = self.agent.handle_message(
                self.db,
                self.project.id,
                "summarize my dataset",
            )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "get_dataset_summary")
        self.assertEqual(response.message, response.reply)
        self.assertEqual(response.tool_result["tool_name"], "get_dataset_summary")
        self.assertEqual(response.tool_result["dataset_id"], self.dataset.id)
        self.assertEqual(response.tool_result["filename"], "pending.csv")
        self.assertEqual(response.tool_result["rows"], 6)
        self.assertIn("yield", response.tool_result["columns"])
        self.assertIn("yield", response.tool_result["numeric_columns"])

    def test_dataset_summary_uses_latest_dataset_id_from_memory(self) -> None:
        newer_dataset = Dataset(
            project_id=self.project.id,
            filename="newer.csv",
            row_count=6,
            column_count=3,
            raw_data_json=json.dumps(
                [
                    {"temperature": 10, "quality": 1, "scrap": 0},
                    {"temperature": 11, "quality": 1, "scrap": 0},
                    {"temperature": 12, "quality": 0, "scrap": 1},
                    {"temperature": 13, "quality": 1, "scrap": 0},
                    {"temperature": 14, "quality": 0, "scrap": 1},
                    {"temperature": 15, "quality": 1, "scrap": 0},
                ]
            ),
        )
        self.db.add(newer_dataset)
        self.db.commit()
        self.db.refresh(newer_dataset)
        upsert_memory(
            self.db,
            self.project.id,
            "latest_dataset_id",
            self.dataset.id,
            memory_type="dataset",
            source="test",
        )

        with patch(
            "app.agent.agent_service.requests.post",
            side_effect=requests.RequestException,
        ):
            response = self.agent.handle_message(
                self.db,
                self.project.id,
                "summarize my dataset",
            )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "get_dataset_summary")
        self.assertEqual(response.tool_result["dataset_id"], self.dataset.id)
        self.assertEqual(response.tool_result["filename"], "pending.csv")
        self.assertIn(
            "I used your latest uploaded dataset: pending.csv.",
            response.reply,
        )

    def test_explicit_dataset_id_overrides_memory_dataset_id(self) -> None:
        explicit_dataset = Dataset(
            project_id=self.project.id,
            filename="explicit.csv",
            row_count=6,
            column_count=3,
            raw_data_json=json.dumps(
                [
                    {"temperature": 10, "quality": 1, "scrap": 0},
                    {"temperature": 11, "quality": 1, "scrap": 0},
                    {"temperature": 12, "quality": 0, "scrap": 1},
                    {"temperature": 13, "quality": 1, "scrap": 0},
                    {"temperature": 14, "quality": 0, "scrap": 1},
                    {"temperature": 15, "quality": 1, "scrap": 0},
                ]
            ),
        )
        self.db.add(explicit_dataset)
        self.db.commit()
        self.db.refresh(explicit_dataset)
        upsert_memory(
            self.db,
            self.project.id,
            "latest_dataset_id",
            self.dataset.id,
            memory_type="dataset",
            source="test",
        )

        with patch(
            "app.agent.agent_service.requests.post",
            side_effect=requests.RequestException,
        ):
            response = self.agent.handle_message(
                self.db,
                self.project.id,
                f"summarize dataset {explicit_dataset.id}",
            )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_result["dataset_id"], explicit_dataset.id)
        self.assertEqual(response.tool_result["filename"], "explicit.csv")
        self.assertNotIn("latest uploaded dataset", response.reply)

    def test_document_question_with_no_documents_returns_clean_message(self) -> None:
        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "what does the uploaded paper say about yield?",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "answer_document_question")
        self.assertEqual(response.reply, NO_DOCUMENTS_MESSAGE)
        self.assertFalse(response.tool_result["success"])
        self.assertEqual(response.tool_result["error"], NO_DOCUMENTS_MESSAGE)

    def test_document_question_response_includes_sources(self) -> None:
        document = Document(
            project_id=self.project.id,
            filename="paper.txt",
            file_path="paper.txt",
            mime_type="text/plain",
            file_size=10,
            extracted_text_path="paper.txt",
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        chunk = DocumentChunk(
            document_id=document.id,
            project_id=self.project.id,
            chunk_index=0,
            content="Yield improved in the reported condition.",
            char_count=len("Yield improved in the reported condition."),
            embedding_json=json.dumps([1.0, 0.0]),
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        fake_answer = SimpleNamespace(
            answer="The paper says yield improved [doc:paper.txt chunk:2].",
            sources=[
                SimpleNamespace(
                    document_id=document.id,
                    filename="paper.txt",
                    chunk_id=7,
                    chunk_index=2,
                    score=0.82,
                    content_preview="Yield improved in the reported condition.",
                )
            ],
        )

        with patch(
            "app.agent.tools.document_tools.generate_document_answer",
            return_value=fake_answer,
        ):
            response = self.agent.handle_message(
                self.db,
                self.project.id,
                "according to the document, what does it mention about yield?",
            )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "answer_document_question")
        self.assertIn("[doc:paper.txt chunk:2]", response.reply)
        self.assertEqual(
            response.tool_result["tool_name"],
            "answer_document_question",
        )
        self.assertEqual(response.tool_result["answer"], fake_answer.answer)
        self.assertEqual(len(response.tool_result["sources"]), 1)
        self.assertEqual(response.tool_result["sources"][0]["chunk_id"], 7)

    def test_user_reply_fills_target_and_clears_pending_action(self) -> None:
        self.agent.handle_message(self.db, self.project.id, "train a model")

        with patch(
            "app.agent.agent_service.requests.post",
            side_effect=requests.RequestException,
        ):
            response = self.agent.handle_message(self.db, self.project.id, "yield")

        self.assertIn("model run #", response.reply)
        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "train_baseline_model")
        self.assertEqual(response.tool_result["tool_name"], "train_baseline_model")
        self.assertIsNotNone(response.tool_result["model_run_id"])
        self.assertEqual(response.tool_result["dataset_id"], self.dataset.id)
        self.assertEqual(response.tool_result["target_column"], "yield")
        self.assertEqual(response.tool_result["task_type"], "regression")
        self.assertIn("r2_score", response.tool_result["metrics"])
        self.assertIn("top_features", response.tool_result)
        self.assertIsNone(get_pending_action(self.db, self.project.id))
        self.assertEqual(self.db.query(ModelRun).count(), 1)
        selected_memory = get_memory(
            self.db,
            self.project.id,
            "selected_target_column",
        )
        self.assertIsNotNone(selected_memory)
        self.assertEqual(selected_memory.value_json, '"yield"')

    def test_train_model_summary_does_not_include_stale_dataset_list_context(self) -> None:
        with patch(
            "app.agent.agent_service.requests.post",
            side_effect=AssertionError("Training summaries should be deterministic."),
        ):
            response = self.agent.handle_message(
                self.db,
                self.project.id,
                "train a model using yield",
            )

        self.assertIn("model run #", response.reply)
        self.assertNotIn("Dataset list is empty", response.reply)
        self.assertNotIn("upload previews", response.reply)

    def test_explain_model_uses_latest_model_run_memory(self) -> None:
        train_response = self.agent.handle_message(
            self.db,
            self.project.id,
            "train a model using yield",
        )
        model_run_id = train_response.tool_result["model_run_id"]

        with patch(
            "app.agent.agent_service.requests.post",
            side_effect=requests.RequestException,
        ):
            response = self.agent.handle_message(
                self.db,
                self.project.id,
                "explain the model",
            )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "explain_latest_model")
        self.assertEqual(response.tool_result["model_run_id"], model_run_id)
        self.assertEqual(response.tool_result["target_column"], "yield")
        self.assertEqual(response.tool_result["task_type"], "regression")
        self.assertIn("metrics", response.tool_result)
        self.assertIn("top_features", response.tool_result)
        self.assertIn(
            "Feature importance indicates predictive association, not necessarily causation.",
            response.reply,
        )

    def test_explain_model_without_model_returns_clean_message(self) -> None:
        self.db.query(ModelRun).delete()
        self.db.commit()

        response = self.agent.handle_message(
            self.db,
            self.project.id,
            "explain latest model",
        )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "explain_latest_model")
        self.assertFalse(response.tool_result["model_available"])
        self.assertEqual(
            response.reply,
            "No trained model is available yet. Train a model first.",
        )

    def test_regular_model_listing_still_works(self) -> None:
        train_response = self.agent.handle_message(
            self.db,
            self.project.id,
            "train a model using yield",
        )

        with patch(
            "app.agent.agent_service.requests.post",
            side_effect=requests.RequestException,
        ):
            response = self.agent.handle_message(
                self.db,
                self.project.id,
                "show previous model runs",
            )

        self.assertTrue(response.tool_used)
        self.assertEqual(response.tool_name, "list_model_runs")
        self.assertIn(f"#{train_response.tool_result['model_run_id']}", response.reply)

    def test_agent_prompt_includes_project_memory_context(self) -> None:
        upsert_memory(
            self.db,
            self.project.id,
            "selected_target_column",
            "yield",
            memory_type="user_decision",
            source="test",
        )

        with patch("app.services.llm_chat.requests.post") as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            mock_post.return_value.json.return_value = {"response": "Use yield."}

            response = self.agent.handle_message(
                self.db,
                self.project.id,
                "what should I remember about this project?",
            )

        prompt = mock_post.call_args.kwargs["json"]["prompt"]
        self.assertEqual(response.reply, "Use yield.")
        self.assertIn("Project memory:", prompt)
        self.assertIn("project_summary", prompt)
        self.assertIn("selected_target_column", prompt)
        self.assertIn("yield", prompt)

    def test_invalid_column_keeps_pending_action_active(self) -> None:
        self.agent.handle_message(self.db, self.project.id, "train a model")

        response = self.agent.handle_message(self.db, self.project.id, "quality")

        self.assertIn("I could not find 'quality'", response.reply)
        self.assertIsNotNone(get_pending_action(self.db, self.project.id))
        self.assertEqual(self.db.query(ModelRun).count(), 0)

    def test_cancel_clears_pending_action(self) -> None:
        self.agent.handle_message(self.db, self.project.id, "train a model")

        response = self.agent.handle_message(self.db, self.project.id, "cancel")

        self.assertEqual(response.reply, "Canceled the pending action.")
        self.assertIsNone(get_pending_action(self.db, self.project.id))


if __name__ == "__main__":
    unittest.main()
