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
from app.agent.tools.document_tools import (
    DOCUMENT_NOT_PROCESSED_MESSAGE,
    EMBEDDINGS_MISSING_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    NO_RELEVANT_CHUNKS_MESSAGE,
    answer_document_question as answer_document_question_tool,
)
from app.agent.tools.model_tools import train_baseline_model
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
from app.db.models import Dataset, Document, DocumentChunk, ModelRun, Project
from app.db.session import get_db
from app.routes.datasets import router as datasets_router
from app.routes.documents import router as documents_router
from app.routes.models import router as models_router
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
                "answer_document_question",
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
