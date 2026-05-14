import json
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import requests
from app.agent.agent_service import AgentService
from app.agent.intent_parser import IntentParser
from app.agent.pending_action import get_pending_action
from app.agent.tools.model_tools import train_baseline_model
from app.agent.tool_executor import ToolExecutor
from app.agent.tool_registry import ToolRegistry
from app.agent.tool_schemas import ToolIntent
from app.db.base import Base
from app.db.models import Dataset, ModelRun, Project
from app.db.session import get_db
from app.routes.datasets import router as datasets_router
from app.routes.models import router as models_router
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
            },
        )


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
        self.assertIn("yield", response.reply)
        pending_action = get_pending_action(self.db, self.project.id)
        self.assertIsNotNone(pending_action)
        self.assertEqual(pending_action.tool_name, "train_baseline_model")

    def test_user_reply_fills_target_and_clears_pending_action(self) -> None:
        self.agent.handle_message(self.db, self.project.id, "train a model")

        with patch(
            "app.agent.agent_service.requests.post",
            side_effect=requests.RequestException,
        ):
            response = self.agent.handle_message(self.db, self.project.id, "yield")

        self.assertIn("model run #", response.reply)
        self.assertIsNone(get_pending_action(self.db, self.project.id))
        self.assertEqual(self.db.query(ModelRun).count(), 1)

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
