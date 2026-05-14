"use client";

import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  BarChart3,
  ChevronDown,
  ChevronUp,
  Files,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Upload,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

const API_BASE_URL = "http://127.0.0.1:8000";

type Project = {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
};

type Dataset = {
  id: number;
  project_id: number;
  filename: string;
  row_count: number;
  column_count: number;
  data: Record<string, unknown>[];
  created_at: string;
};

type DatasetUploadPreview = {
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  preview: Record<string, unknown>[];
  data: Record<string, unknown>[];
  profile: {
    missing_values: Record<string, number>;
    column_types: Record<string, string>;
  };
};

type DatasetUploadResult = {
  dataset: Dataset;
  preview: DatasetUploadPreview;
};

type FeatureImportance = {
  feature: string;
  importance: number;
};

type ModelTrainingResult = {
  model_run_id?: number | null;
  project_id?: number | null;
  dataset_id?: number | null;
  problem_type: string;
  task_type?: string | null;
  model_type: string;
  target_column?: string | null;
  metrics: Record<string, number>;
  feature_importance: FeatureImportance[];
};

type ModelRun = {
  id: number;
  project_id: number;
  dataset_id: number | null;
  target_column: string;
  task_type: string;
  model_type: string;
  metrics: Record<string, number>;
  feature_importance: FeatureImportance[];
  created_at: string;
};

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

type DatasetChatSummary = {
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  preview: Record<string, unknown>[];
  profile: DatasetUploadPreview["profile"];
};

type WorkbenchTab = "upload" | "files" | "model";

export default function Home() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<number | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const [selectedCsvFile, setSelectedCsvFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] =
    useState<DatasetUploadPreview | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [modelTrainingResult, setModelTrainingResult] =
    useState<ModelTrainingResult | null>(null);
  const [modelRuns, setModelRuns] = useState<ModelRun[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [activeWorkbenchTab, setActiveWorkbenchTab] =
    useState<WorkbenchTab>("upload");
  const [isWorkspacePanelOpen, setIsWorkspacePanelOpen] = useState(true);
  const [isWorkbenchPanelOpen, setIsWorkbenchPanelOpen] = useState(true);
  const [isCopilotPanelOpen, setIsCopilotPanelOpen] = useState(true);
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [isUploadingCsv, setIsUploadingCsv] = useState(false);
  const [isTrainingModel, setIsTrainingModel] = useState(false);
  const [isLoadingModelRuns, setIsLoadingModelRuns] = useState(false);
  const [isSendingChatMessage, setIsSendingChatMessage] = useState(false);
  const [deletingProjectId, setDeletingProjectId] = useState<number | null>(
    null,
  );
  const [deletingDatasetId, setDeletingDatasetId] = useState<number | null>(
    null,
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const activeProject = useMemo(
    () => projects.find((project) => project.id === activeProjectId) ?? null,
    [activeProjectId, projects],
  );

  const latestDataset = datasets[0] ?? null;

  const modelDataset = useMemo(() => {
    if (latestDataset) {
      const firstRow = latestDataset.data[0] ?? {};
      return {
        datasetId: latestDataset.id,
        filename: latestDataset.filename,
        rows: latestDataset.row_count,
        columns: latestDataset.column_count,
        columnNames: Object.keys(firstRow),
        data: latestDataset.data,
      };
    }

    if (!uploadPreview) {
      return null;
    }

    return {
      datasetId: null,
      filename: uploadPreview.filename,
      rows: uploadPreview.rows,
      columns: uploadPreview.columns,
      columnNames: uploadPreview.column_names,
      data: uploadPreview.data,
    };
  }, [latestDataset, uploadPreview]);

  const datasetChatSummary = useMemo<DatasetChatSummary | null>(() => {
    if (!uploadPreview) {
      return null;
    }

    return {
      filename: uploadPreview.filename,
      rows: uploadPreview.rows,
      columns: uploadPreview.columns,
      column_names: uploadPreview.column_names,
      preview: uploadPreview.preview,
      profile: uploadPreview.profile,
    };
  }, [uploadPreview]);

  useEffect(() => {
    if (!modelDataset) {
      setTargetColumn("");
      return;
    }

    setTargetColumn((currentTargetColumn) => {
      if (
        currentTargetColumn &&
        modelDataset.columnNames.includes(currentTargetColumn)
      ) {
        return currentTargetColumn;
      }
      return modelDataset.columnNames[0] ?? "";
    });
  }, [modelDataset]);

  const loadDatasets = useCallback(async (projectId: number) => {
    try {
      setIsLoadingDatasets(true);
      setErrorMessage(null);

      const response = await fetch(
        `${API_BASE_URL}/projects/${projectId}/datasets`,
      );
      if (!response.ok) {
        throw new Error("Could not load datasets.");
      }

      setDatasets((await response.json()) as Dataset[]);
    } catch (error) {
      setDatasets([]);
      setErrorMessage(
        error instanceof Error ? error.message : "Could not load datasets.",
      );
    } finally {
      setIsLoadingDatasets(false);
    }
  }, []);

  const loadChatHistory = useCallback(async (projectId: number) => {
    try {
      const response = await fetch(`${API_BASE_URL}/projects/${projectId}/chat`);
      if (!response.ok) {
        throw new Error("Could not load chat history.");
      }

      setChatMessages((await response.json()) as ChatMessage[]);
    } catch (error) {
      setChatMessages([]);
      setErrorMessage(
        error instanceof Error ? error.message : "Could not load chat history.",
      );
    }
  }, []);

  const loadModelRuns = useCallback(async (projectId: number) => {
    try {
      setIsLoadingModelRuns(true);
      const response = await fetch(
        `${API_BASE_URL}/models/projects/${projectId}/model-runs`,
      );
      if (!response.ok) {
        throw new Error("Could not load model runs.");
      }

      setModelRuns((await response.json()) as ModelRun[]);
    } catch (error) {
      setModelRuns([]);
      setErrorMessage(
        error instanceof Error ? error.message : "Could not load model runs.",
      );
    } finally {
      setIsLoadingModelRuns(false);
    }
  }, []);

  useEffect(() => {
    async function loadProjects() {
      try {
        setIsLoadingProjects(true);
        setErrorMessage(null);

        const response = await fetch(`${API_BASE_URL}/projects`);
        if (!response.ok) {
          throw new Error("Could not load projects.");
        }

        const projectList = (await response.json()) as Project[];
        setProjects(projectList);
        setActiveProjectId((currentProjectId) => {
          if (
            currentProjectId &&
            projectList.some((project) => project.id === currentProjectId)
          ) {
            return currentProjectId;
          }
          return projectList[0]?.id ?? null;
        });
      } catch (error) {
        setErrorMessage(
          error instanceof Error ? error.message : "Could not load projects.",
        );
      } finally {
        setIsLoadingProjects(false);
      }
    }

    loadProjects();
  }, []);

  useEffect(() => {
    setUploadPreview(null);
    setSelectedCsvFile(null);
    setTargetColumn("");
    setModelTrainingResult(null);
    setModelRuns([]);
    setChatMessages([]);
    setChatInput("");
    setSuccessMessage(null);

    if (activeProjectId === null) {
      setDatasets([]);
      return;
    }

    loadDatasets(activeProjectId);
    loadChatHistory(activeProjectId);
    loadModelRuns(activeProjectId);
  }, [activeProjectId, loadChatHistory, loadDatasets, loadModelRuns]);

  async function handleCreateProject(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedName = projectName.trim();
    const trimmedDescription = projectDescription.trim();
    if (!trimmedName) {
      setErrorMessage("Project name is required.");
      return;
    }

    try {
      setIsCreatingProject(true);
      setErrorMessage(null);
      setSuccessMessage(null);

      const response = await fetch(`${API_BASE_URL}/projects`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: trimmedName,
          description: trimmedDescription || null,
        }),
      });

      if (!response.ok) {
        throw new Error("Could not create project.");
      }

      const createdProject = (await response.json()) as Project;
      setProjects((currentProjects) => [createdProject, ...currentProjects]);
      setActiveProjectId(createdProject.id);
      setProjectName("");
      setProjectDescription("");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not create project.",
      );
    } finally {
      setIsCreatingProject(false);
    }
  }

  async function handleDeleteProject(project: Project) {
    const confirmed = window.confirm(
      `Delete project "${project.name}" and all of its datasets and chat history?`,
    );
    if (!confirmed) {
      return;
    }

    try {
      setDeletingProjectId(project.id);
      setErrorMessage(null);
      setSuccessMessage(null);

      const response = await fetch(`${API_BASE_URL}/projects/${project.id}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Could not delete project.");
      }

      const remainingProjects = projects.filter(
        (currentProject) => currentProject.id !== project.id,
      );
      setProjects(remainingProjects);
      if (activeProjectId === project.id) {
        setActiveProjectId(remainingProjects[0]?.id ?? null);
      }
      setSuccessMessage("Project deleted.");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not delete project.",
      );
    } finally {
      setDeletingProjectId(null);
    }
  }

  async function handleUploadPreview(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!activeProjectId) {
      setErrorMessage("Create or select a project before uploading a dataset.");
      return;
    }

    if (!selectedCsvFile) {
      setErrorMessage("Choose a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedCsvFile);

    try {
      setIsUploadingCsv(true);
      setErrorMessage(null);
      setSuccessMessage(null);

      const response = await fetch(
        `${API_BASE_URL}/projects/${activeProjectId}/datasets/upload`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        const errorBody = (await response.json().catch(() => null)) as {
          detail?: string;
        } | null;
        throw new Error(errorBody?.detail || "Could not upload CSV.");
      }

      const result = (await response.json()) as DatasetUploadResult;
      setUploadPreview(result.preview);
      setTargetColumn(result.preview.column_names[0] ?? "");
      setModelTrainingResult(null);
      setSuccessMessage(null);
      setSelectedCsvFile(null);
      setActiveWorkbenchTab("files");
      await loadDatasets(activeProjectId);
    } catch (error) {
      setUploadPreview(null);
      setTargetColumn("");
      setModelTrainingResult(null);
      setErrorMessage(
        error instanceof Error ? error.message : "Could not upload CSV.",
      );
    } finally {
      setIsUploadingCsv(false);
    }
  }

  async function handleDeleteDataset(dataset: Dataset) {
    if (!activeProjectId) {
      return;
    }

    const confirmed = window.confirm(`Delete dataset "${dataset.filename}"?`);
    if (!confirmed) {
      return;
    }

    try {
      setDeletingDatasetId(dataset.id);
      setErrorMessage(null);
      setSuccessMessage(null);

      const response = await fetch(
        `${API_BASE_URL}/projects/${activeProjectId}/datasets/${dataset.id}`,
        {
          method: "DELETE",
        },
      );

      if (!response.ok) {
        throw new Error("Could not delete dataset.");
      }

      setDatasets((currentDatasets) =>
        currentDatasets.filter(
          (currentDataset) => currentDataset.id !== dataset.id,
        ),
      );
      setModelTrainingResult(null);
      setSuccessMessage("Dataset deleted.");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not delete dataset.",
      );
    } finally {
      setDeletingDatasetId(null);
    }
  }

  async function handleTrainModel() {
    if (!modelDataset || !targetColumn || !activeProjectId) {
      setErrorMessage("Choose a target column before training.");
      return;
    }

    if (!modelDataset.datasetId) {
      setErrorMessage("Refresh the Files tab before training this dataset.");
      return;
    }

    try {
      setIsTrainingModel(true);
      setErrorMessage(null);
      setSuccessMessage(null);
      setModelTrainingResult(null);

      const response = await fetch(`${API_BASE_URL}/models/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          project_id: activeProjectId,
          dataset_id: modelDataset.datasetId,
          target_column: targetColumn,
        }),
      });

      if (!response.ok) {
        const errorBody = (await response.json().catch(() => null)) as {
          detail?: string;
        } | null;
        throw new Error(errorBody?.detail || "Could not train model.");
      }

      const result = (await response.json()) as ModelTrainingResult;
      setModelTrainingResult(result);
      setSuccessMessage(
        `Model run #${result.model_run_id ?? ""} saved to workspace.`,
      );
      await loadModelRuns(activeProjectId);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not train model.",
      );
    } finally {
      setIsTrainingModel(false);
    }
  }

  async function handleSendChatMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedMessage = chatInput.trim();
    if (!trimmedMessage || isSendingChatMessage || activeProjectId === null) {
      return;
    }
    setChatInput("");
    setIsSendingChatMessage(true);

    try {
      setChatMessages((currentMessages) => [
        ...currentMessages,
        {
          id: Date.now(),
          role: "user",
          created_at: new Date().toISOString(),
          content: trimmedMessage,
        },
      ]);

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: trimmedMessage,
          project_id: activeProjectId,
          dataset_summary: datasetChatSummary,
          model_result: modelTrainingResult,
        }),
      });

      if (!response.ok) {
        throw new Error("The copilot could not respond right now.");
      }

      const result = (await response.json()) as { reply?: string };
      const assistantReply =
        result.reply?.trim() ||
        "I did not receive a usable reply from the copilot.";
      setChatMessages((currentMessages) => [
        ...currentMessages,
        {
          id: Date.now() + 1,
          role: "assistant",
          created_at: new Date().toISOString(),
          content: assistantReply,
        },
      ]);
      await loadChatHistory(activeProjectId);
      await loadModelRuns(activeProjectId);
    } catch (error) {
      setChatMessages((currentMessages) => [
        ...currentMessages,
        {
          id: Date.now() + 1,
          role: "assistant",
          created_at: new Date().toISOString(),
          content:
            error instanceof Error
              ? error.message
              : "The copilot could not respond right now.",
        },
      ]);
    } finally {
      setIsSendingChatMessage(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#090d12] text-zinc-100">
      <div className="flex min-h-screen flex-col lg:flex-row">
        {isWorkspacePanelOpen ? (
        <aside className="border-b border-zinc-800 bg-[#0d1219] shadow-2xl shadow-black/30 lg:w-80 lg:border-b-0 lg:border-r">
          <div className="border-b border-zinc-800 px-6 py-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-medium text-zinc-500">Workspace</p>
                <h1 className="mt-2 text-2xl font-semibold">
                  AI Scientist Copilot
                </h1>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label="Collapse workspace panel"
                title="Collapse workspace panel"
                onClick={() => setIsWorkspacePanelOpen(false)}
              >
                <PanelLeftClose className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <form
            onSubmit={handleCreateProject}
            className="space-y-3 border-b border-zinc-800 px-6 py-5"
          >
            <div>
              <label
                htmlFor="project-name"
                className="text-sm font-medium text-zinc-300"
              >
                Project name
              </label>
              <Input
                id="project-name"
                value={projectName}
                onChange={(event) => setProjectName(event.target.value)}
                className="mt-1"
                placeholder="Battery electrolyte study"
              />
            </div>

            <div>
              <label
                htmlFor="project-description"
                className="text-sm font-medium text-zinc-300"
              >
                Description
              </label>
              <Textarea
                id="project-description"
                value={projectDescription}
                onChange={(event) =>
                  setProjectDescription(event.target.value)
                }
                className="mt-1 resize-none"
                placeholder="Optional project context"
              />
            </div>

            <Button
              type="submit"
              disabled={isCreatingProject}
              className="w-full"
            >
              {isCreatingProject ? "Creating..." : "Create Project"}
            </Button>
          </form>

          <section className="px-3 py-4">
            <div className="mb-2 flex items-center justify-between px-3">
              <h2 className="text-sm font-semibold text-zinc-300">
                Projects
              </h2>
              <span className="text-xs text-zinc-500">{projects.length}</span>
            </div>

            {isLoadingProjects ? (
              <p className="px-3 py-4 text-sm text-zinc-500">
                Loading projects...
              </p>
            ) : projects.length === 0 ? (
              <p className="px-3 py-4 text-sm leading-6 text-zinc-500">
                Create a project to start building a scientific workspace.
              </p>
            ) : (
              <div className="space-y-1">
                {projects.map((project) => {
                  const isActive = project.id === activeProjectId;
                  return (
                    <div
                      key={project.id}
                      className={`project-tab ${
                        isActive ? "project-tab-active" : ""
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        <button
                          type="button"
                          onClick={() => setActiveProjectId(project.id)}
                          className="min-w-0 flex-1 text-left"
                        >
                          <span className="block truncate text-sm font-semibold">
                            {project.name}
                          </span>
                          <span className="mt-1 block truncate text-xs text-zinc-500">
                            {project.description || "No description"}
                          </span>
                        </button>
                        <IconDeleteButton
                          ariaLabel={`Delete project ${project.name}`}
                          disabled={deletingProjectId === project.id}
                          onClick={() => handleDeleteProject(project)}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </aside>
        ) : (
          <aside className="border-b border-zinc-800 bg-[#0d1219] px-2 py-3 lg:border-b-0 lg:border-r">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Expand workspace panel"
              title="Expand workspace panel"
              onClick={() => setIsWorkspacePanelOpen(true)}
            >
              <PanelLeftOpen className="h-4 w-4" />
            </Button>
          </aside>
        )}

        <section className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
          {errorMessage ? (
            <div className="mb-5 rounded-md border border-red-500/30 bg-red-950/40 px-4 py-3 text-sm text-red-200 shadow-lg shadow-black/20">
              {errorMessage}
            </div>
          ) : null}

          {successMessage ? (
            <div className="mb-5 rounded-md border border-emerald-500/30 bg-emerald-950/40 px-4 py-3 text-sm text-emerald-200 shadow-lg shadow-black/20">
              {successMessage}
            </div>
          ) : null}

          {activeProject ? (
            <div className="space-y-6">
              <Card>
                <CardHeader className="flex flex-row items-start justify-between gap-4">
                  <div>
                    <p className="text-sm font-medium text-zinc-500">
                      Active project
                    </p>
                    <CardTitle className="mt-2 text-3xl">
                      {activeProject.name}
                    </CardTitle>
                    {activeProject.description ? (
                      <p className="mt-3 max-w-3xl text-sm leading-6 text-zinc-400">
                        {activeProject.description}
                      </p>
                    ) : null}
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    aria-label={
                      isWorkbenchPanelOpen
                        ? "Collapse workbench"
                        : "Expand workbench"
                    }
                    title={
                      isWorkbenchPanelOpen
                        ? "Collapse workbench"
                        : "Expand workbench"
                    }
                    onClick={() =>
                      setIsWorkbenchPanelOpen((isOpen) => !isOpen)
                    }
                  >
                    {isWorkbenchPanelOpen ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </Button>
                </CardHeader>

                {isWorkbenchPanelOpen ? (
                  <div className="space-y-5 px-5 py-5">
                    <div className="grid gap-4 sm:grid-cols-3">
                      <Metric
                        label="Files"
                        value={datasets.length.toString()}
                      />
                      <Metric
                        label="Rows"
                        value={datasets
                          .reduce(
                            (total, dataset) => total + dataset.row_count,
                            0,
                          )
                          .toLocaleString()}
                      />
                      <Metric
                        label="Columns"
                        value={datasets
                          .reduce(
                            (total, dataset) => total + dataset.column_count,
                            0,
                          )
                          .toLocaleString()}
                      />
                    </div>

                    <WorkbenchTabs
                      activeTab={activeWorkbenchTab}
                      datasetCount={datasets.length}
                      onTabChange={setActiveWorkbenchTab}
                    />

                    {activeWorkbenchTab === "upload" ? (
                      <CsvUploadPanel
                        isUploadingCsv={isUploadingCsv}
                        onFileChange={(file) => {
                          setSelectedCsvFile(file);
                          setUploadPreview(null);
                          setTargetColumn("");
                          setModelTrainingResult(null);
                          setSuccessMessage(null);
                        }}
                        onUploadPreview={handleUploadPreview}
                        selectedCsvFile={selectedCsvFile}
                        uploadPreview={uploadPreview}
                      />
                    ) : null}

                    {activeWorkbenchTab === "files" ? (
                      <DatasetFilesPanel
                        datasets={datasets}
                        deletingDatasetId={deletingDatasetId}
                        isLoadingDatasets={isLoadingDatasets}
                        onDeleteDataset={handleDeleteDataset}
                      />
                    ) : null}

                    {activeWorkbenchTab === "model" ? (
                      <ModelWorkbenchPanel
                        isTrainingModel={isTrainingModel}
                        isLoadingModelRuns={isLoadingModelRuns}
                        modelDataset={modelDataset}
                        modelRuns={modelRuns}
                        modelTrainingResult={modelTrainingResult}
                        onTargetColumnChange={(columnName) => {
                          setTargetColumn(columnName);
                          setModelTrainingResult(null);
                        }}
                        onTrainModel={handleTrainModel}
                        targetColumn={targetColumn}
                      />
                    ) : null}
                  </div>
                ) : null}
              </Card>
            </div>
          ) : (
            <div className="flex min-h-[60vh] items-center justify-center">
              <div className="max-w-lg text-center">
                <p className="text-sm font-medium text-zinc-500">
                  AI Scientist Copilot
                </p>
                <h2 className="mt-3 text-3xl font-semibold">
                  Create or select a project
                </h2>
                <p className="mt-3 text-sm leading-6 text-zinc-400">
                  Projects organize datasets, analysis, models, and future
                  copilot memory for each scientific workspace.
                </p>
              </div>
            </div>
          )}
        </section>

        {isCopilotPanelOpen ? (
          <CopilotChatPanel
            activeProjectId={activeProjectId}
            chatInput={chatInput}
            datasetSummary={datasetChatSummary}
            isSending={isSendingChatMessage}
            messages={chatMessages}
            modelResult={modelTrainingResult}
            onCollapse={() => setIsCopilotPanelOpen(false)}
            onInputChange={setChatInput}
            onSend={handleSendChatMessage}
          />
        ) : (
          <aside className="border-t border-zinc-800 bg-[#0d1219] px-2 py-3 lg:border-l lg:border-t-0">
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Expand copilot panel"
              title="Expand copilot panel"
              onClick={() => setIsCopilotPanelOpen(true)}
            >
              <PanelRightOpen className="h-4 w-4" />
            </Button>
          </aside>
        )}
      </div>
    </main>
  );
}

function CopilotChatPanel({
  activeProjectId,
  chatInput,
  datasetSummary,
  isSending,
  messages,
  modelResult,
  onCollapse,
  onInputChange,
  onSend,
}: {
  activeProjectId: number | null;
  chatInput: string;
  datasetSummary: DatasetChatSummary | null;
  isSending: boolean;
  messages: ChatMessage[];
  modelResult: ModelTrainingResult | null;
  onCollapse: () => void;
  onInputChange: (value: string) => void;
  onSend: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages, isSending]);

  return (
    <aside className="border-t border-zinc-800 bg-[#0d1219] shadow-2xl shadow-black/30 lg:sticky lg:top-0 lg:flex lg:h-screen lg:w-96 lg:flex-col lg:border-l lg:border-t-0">
      <div className="shrink-0 border-b border-zinc-800 px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-medium text-zinc-500">
              Copilot
            </p>
            <h2 className="mt-1 text-lg font-semibold">
              Scientist chat
            </h2>
          </div>
          <div className="flex gap-1.5 pt-1">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                datasetSummary ? "bg-emerald-400" : "bg-zinc-700"
              }`}
              title={
                datasetSummary
                  ? "Dataset preview is included"
                  : "No dataset preview available"
              }
            />
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                modelResult ? "bg-cyan-400" : "bg-zinc-700"
              }`}
              title={
                modelResult
                  ? "Model result is included"
                  : "No model result available"
              }
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label="Collapse copilot panel"
              title="Collapse copilot panel"
              onClick={onCollapse}
              className="-mt-2 ml-1"
            >
              <PanelRightClose className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <div className="flex max-h-[28rem] flex-1 flex-col gap-3 overflow-y-auto px-5 py-4 lg:min-h-0 lg:max-h-none">
        {messages.length === 0 ? (
          <div className="rounded-md border border-dashed border-zinc-700 bg-[#0f151d] px-4 py-5 text-sm leading-6 text-zinc-400 shadow-lg shadow-black/20">
            Ask about the uploaded dataset, model metrics, feature importance,
            or next experiment ideas. Current context is included when
            available.
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[85%] rounded-md px-3 py-2 text-sm leading-6 shadow-sm ${
                  message.role === "user"
                    ? "bg-[#1f6feb] text-white"
                    : "border border-zinc-800 bg-[#131b26] text-zinc-200"
                }`}
              >
                <p className="whitespace-pre-wrap break-words">
                  {message.content}
                </p>
              </div>
            </div>
          ))
        )}

        {isSending ? (
          <div className="flex justify-start">
            <div className="rounded-md border border-zinc-800 bg-[#131b26] px-3 py-2 text-sm text-zinc-500 shadow-sm">
              Thinking...
            </div>
          </div>
        ) : null}
        <div ref={messagesEndRef} />
      </div>

      <form
        onSubmit={onSend}
        className="shrink-0 border-t border-zinc-800 bg-[#0b1017] p-4"
      >
        <label htmlFor="copilot-message" className="sr-only">
          Copilot message
        </label>
        <Textarea
          id="copilot-message"
          value={chatInput}
          onChange={(event) => onInputChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          className="min-h-24 resize-none"
          placeholder="Ask the copilot about this workspace..."
        />
        <Button
          type="submit"
          disabled={!activeProjectId || !chatInput.trim() || isSending}
          className="mt-3 w-full"
        >
          {isSending ? "Sending..." : "Send"}
        </Button>
      </form>
    </aside>
  );
}

function CsvUploadPanel({
  isUploadingCsv,
  onFileChange,
  onUploadPreview,
  selectedCsvFile,
  uploadPreview,
}: {
  isUploadingCsv: boolean;
  onFileChange: (file: File | null) => void;
  onUploadPreview: (event: FormEvent<HTMLFormElement>) => void;
  selectedCsvFile: File | null;
  uploadPreview: DatasetUploadPreview | null;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>CSV upload</CardTitle>
      </CardHeader>

      <form
        onSubmit={onUploadPreview}
        className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center"
      >
        <Input
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
          className="text-zinc-300 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-800 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-zinc-200 hover:file:bg-zinc-700"
        />
        <Button
          type="submit"
          disabled={!selectedCsvFile || isUploadingCsv}
          variant="secondary"
        >
          {isUploadingCsv ? "Uploading..." : "Upload CSV"}
        </Button>
      </form>

      {uploadPreview ? (
        <div className="border-t border-zinc-800 px-5 py-5">
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h4 className="text-base font-semibold">
                {uploadPreview.filename}
              </h4>
              <p className="mt-1 text-sm text-zinc-500">
                {uploadPreview.rows.toLocaleString()} rows,{" "}
                {uploadPreview.columns.toLocaleString()} columns
              </p>
            </div>
            <Badge>Saved to project</Badge>
          </div>

          <div className="mb-4 flex flex-wrap gap-2">
            {uploadPreview.column_names.map((columnName) => (
              <Badge key={columnName}>
                {columnName}
              </Badge>
            ))}
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <ProfileTable
              missingValues={uploadPreview.profile.missing_values}
              columnTypes={uploadPreview.profile.column_types}
            />
            <PreviewTable
              columnNames={uploadPreview.column_names}
              rows={uploadPreview.preview}
            />
          </div>
        </div>
      ) : null}
    </Card>
  );
}

function WorkbenchTabs({
  activeTab,
  datasetCount,
  onTabChange,
}: {
  activeTab: WorkbenchTab;
  datasetCount: number;
  onTabChange: (tab: WorkbenchTab) => void;
}) {
  const tabs: Array<{
    id: WorkbenchTab;
    label: string;
    icon: typeof Upload;
    count?: number;
  }> = [
    { id: "upload", label: "Upload", icon: Upload },
    { id: "files", label: "Files", icon: Files, count: datasetCount },
    { id: "model", label: "Model", icon: BarChart3 },
  ];

  return (
    <div className="flex flex-wrap gap-2 border-b border-zinc-800 pb-3">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;
        return (
          <Button
            key={tab.id}
            type="button"
            variant={isActive ? "default" : "ghost"}
            size="sm"
            onClick={() => onTabChange(tab.id)}
            className="gap-2"
          >
            <Icon className="h-4 w-4" />
            {tab.label}
            {typeof tab.count === "number" ? (
              <span className="rounded bg-black/15 px-1.5 text-xs">
                {tab.count}
              </span>
            ) : null}
          </Button>
        );
      })}
    </div>
  );
}

function DatasetFilesPanel({
  datasets,
  deletingDatasetId,
  isLoadingDatasets,
  onDeleteDataset,
}: {
  datasets: Dataset[];
  deletingDatasetId: number | null;
  isLoadingDatasets: boolean;
  onDeleteDataset: (dataset: Dataset) => void;
}) {
  if (datasets.length === 0 && !isLoadingDatasets) {
    return (
      <Card className="border-dashed px-5 py-10 text-center">
        <h4 className="text-base font-semibold text-zinc-100">
          No files uploaded yet
        </h4>
        <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-zinc-500">
          Upload a CSV from the Upload tab. It will appear here automatically.
        </p>
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden">
      <div className="grid grid-cols-12 border-b border-zinc-800 bg-[#131b26] px-4 py-3 text-xs font-medium text-zinc-500">
        <span className="col-span-4">Filename</span>
        <span className="col-span-2 text-right">Rows</span>
        <span className="col-span-2 text-right">Columns</span>
        <span className="col-span-2 text-right">Uploaded</span>
        <span className="col-span-2 text-right">Action</span>
      </div>

      {isLoadingDatasets ? (
        <div className="px-4 py-4 text-sm text-zinc-500">Loading files...</div>
      ) : (
        datasets.map((dataset) => (
          <div
            key={dataset.id}
            className="grid grid-cols-12 items-center border-b border-zinc-800/80 px-4 py-4 text-sm transition hover:bg-[#151f2b] last:border-b-0"
          >
            <span className="col-span-4 truncate font-medium text-zinc-100">
              {dataset.filename}
            </span>
            <span className="col-span-2 text-right text-zinc-400">
              {dataset.row_count.toLocaleString()}
            </span>
            <span className="col-span-2 text-right text-zinc-400">
              {dataset.column_count.toLocaleString()}
            </span>
            <span className="col-span-2 text-right text-zinc-500">
              {formatDate(dataset.created_at)}
            </span>
            <span className="col-span-2 text-right">
              <IconDeleteButton
                ariaLabel={`Delete dataset ${dataset.filename}`}
                disabled={deletingDatasetId === dataset.id}
                onClick={() => onDeleteDataset(dataset)}
              />
            </span>
          </div>
        ))
      )}
    </Card>
  );
}

function ModelWorkbenchPanel({
  isTrainingModel,
  isLoadingModelRuns,
  modelDataset,
  modelRuns,
  modelTrainingResult,
  onTargetColumnChange,
  onTrainModel,
  targetColumn,
}: {
  isTrainingModel: boolean;
  isLoadingModelRuns: boolean;
  modelDataset: {
    datasetId: number | null;
    filename: string;
    rows: number;
    columns: number;
    columnNames: string[];
    data: Record<string, unknown>[];
  } | null;
  modelRuns: ModelRun[];
  modelTrainingResult: ModelTrainingResult | null;
  onTargetColumnChange: (columnName: string) => void;
  onTrainModel: () => void;
  targetColumn: string;
}) {
  if (!modelDataset) {
    return (
      <Card className="border-dashed px-5 py-10 text-center">
        <h4 className="text-base font-semibold text-zinc-100">
          No project files available
        </h4>
        <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-zinc-500">
          Upload a CSV in the Upload tab. Saved files become available for
          modeling automatically.
        </p>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Baseline model</CardTitle>
      </CardHeader>
      <div className="space-y-4 px-5 py-5">
        <div>
          <h4 className="text-base font-semibold">{modelDataset.filename}</h4>
          <p className="mt-1 text-sm text-zinc-500">
            {modelDataset.rows.toLocaleString()} rows,{" "}
            {modelDataset.columns.toLocaleString()} columns
          </p>
        </div>
        <div className="rounded-md border border-zinc-800 bg-[#0b1017] px-4 py-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="flex-1">
              <label
                htmlFor="target-column"
                className="text-sm font-medium text-zinc-300"
              >
                Target column
              </label>
              <select
                id="target-column"
                value={targetColumn}
                onChange={(event) => onTargetColumnChange(event.target.value)}
                className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground outline-none transition focus-visible:ring-2 focus-visible:ring-ring"
              >
                {modelDataset.columnNames.map((columnName) => (
                  <option key={columnName} value={columnName}>
                    {columnName}
                  </option>
                ))}
              </select>
            </div>

            <Button
              type="button"
              onClick={onTrainModel}
              disabled={!targetColumn || isTrainingModel}
              variant="secondary"
            >
              {isTrainingModel ? "Training..." : "Train Model"}
            </Button>
          </div>
        </div>

        {modelTrainingResult ? (
          <ModelTrainingResults result={modelTrainingResult} />
        ) : null}

        <ModelRunsList
          isLoading={isLoadingModelRuns}
          modelRuns={modelRuns}
        />
      </div>
    </Card>
  );
}

function ModelRunsList({
  isLoading,
  modelRuns,
}: {
  isLoading: boolean;
  modelRuns: ModelRun[];
}) {
  return (
    <Card className="overflow-hidden">
      <div className="surface-header">Model runs</div>
      {isLoading ? (
        <p className="px-4 py-4 text-sm text-zinc-500">Loading model runs...</p>
      ) : modelRuns.length === 0 ? (
        <p className="px-4 py-4 text-sm text-zinc-500">
          No model runs saved yet.
        </p>
      ) : (
        modelRuns.map((modelRun) => (
          <div
            key={modelRun.id}
            className="grid grid-cols-12 items-center border-t border-zinc-800 px-4 py-3 text-sm"
          >
            <span className="col-span-2 font-medium text-zinc-100">
              #{modelRun.id}
            </span>
            <span className="col-span-3 truncate text-zinc-300">
              {modelRun.model_type}
            </span>
            <span className="col-span-3 text-zinc-400">
              {modelRun.task_type}
            </span>
            <span className="col-span-4 truncate text-right text-zinc-400">
              target: {modelRun.target_column}
            </span>
          </div>
        ))
      )}
    </Card>
  );
}

function ModelTrainingResults({ result }: { result: ModelTrainingResult }) {
  const topFeatureImportance = result.feature_importance.slice(0, 10);

  return (
    <div className="mb-4 grid gap-4 lg:grid-cols-2">
      <div className="rounded-md border border-zinc-800 bg-[#131b26] px-4 py-4 shadow-lg shadow-black/20">
        <h4 className="text-sm font-semibold text-zinc-100">
          Model summary
        </h4>
        <dl className="mt-4 grid gap-3 text-sm">
          <div className="flex items-center justify-between gap-4">
            <dt className="text-zinc-500">Problem type</dt>
            <dd className="font-medium capitalize text-zinc-100">
              {result.problem_type}
            </dd>
          </div>
          <div className="flex items-center justify-between gap-4">
            <dt className="text-zinc-500">Model type</dt>
            <dd className="font-medium text-zinc-100">
              {result.model_type}
            </dd>
          </div>
        </dl>
      </div>

      <div className="rounded-md border border-zinc-800 bg-[#131b26] px-4 py-4 shadow-lg shadow-black/20">
        <h4 className="text-sm font-semibold text-zinc-100">Metrics</h4>
        <dl className="mt-4 grid gap-3 text-sm">
          {Object.entries(result.metrics).map(([metricName, metricValue]) => (
            <div
              key={metricName}
              className="flex items-center justify-between gap-4"
            >
              <dt className="text-zinc-500">{formatMetricName(metricName)}</dt>
              <dd className="font-medium tabular-nums text-zinc-100">
                {formatMetricValue(metricValue)}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="surface overflow-hidden lg:col-span-2">
        <div className="surface-header">
          Top feature importance
        </div>
        {topFeatureImportance.length === 0 ? (
          <p className="px-4 py-4 text-sm text-zinc-500">
            No feature importance values returned.
          </p>
        ) : (
          <div className="max-h-80 overflow-auto">
            <table className="w-full min-w-[32rem] text-left text-sm">
              <thead className="bg-[#131b26] text-xs font-medium text-zinc-500">
                <tr>
                  <th className="px-4 py-3">Feature</th>
                  <th className="px-4 py-3 text-right">Importance</th>
                </tr>
              </thead>
              <tbody>
                {topFeatureImportance.map((item) => (
                  <tr key={item.feature} className="border-t border-zinc-800">
                    <td className="max-w-96 truncate px-4 py-3 font-medium text-zinc-100">
                      {item.feature}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-zinc-400">
                      {formatMetricValue(item.importance)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function ProfileTable({
  columnTypes,
  missingValues,
}: {
  columnTypes: Record<string, string>;
  missingValues: Record<string, number>;
}) {
  return (
    <div className="surface overflow-hidden">
      <div className="surface-header">
        Profile
      </div>
      <div className="max-h-80 overflow-auto">
        <table className="w-full min-w-96 text-left text-sm">
          <thead className="bg-[#131b26] text-xs font-medium text-zinc-500">
            <tr>
              <th className="px-4 py-3">Column</th>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3 text-right">Missing</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(columnTypes).map(([columnName, columnType]) => (
              <tr key={columnName} className="border-t border-zinc-800">
                <td className="px-4 py-3 font-medium text-zinc-100">
                  {columnName}
                </td>
                <td className="px-4 py-3 text-zinc-400">{columnType}</td>
                <td className="px-4 py-3 text-right text-zinc-400">
                  {(missingValues[columnName] ?? 0).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PreviewTable({
  columnNames,
  rows,
}: {
  columnNames: string[];
  rows: Record<string, unknown>[];
}) {
  return (
    <div className="surface overflow-hidden">
      <div className="surface-header">
        Preview
      </div>
      <div className="max-h-80 overflow-auto">
        <table className="w-full min-w-[36rem] text-left text-sm">
          <thead className="bg-[#131b26] text-xs font-medium text-zinc-500">
            <tr>
              {columnNames.map((columnName) => (
                <th key={columnName} className="px-4 py-3">
                  {columnName}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-t border-zinc-800">
                {columnNames.map((columnName) => (
                  <td
                    key={columnName}
                    className="max-w-44 truncate px-4 py-3 text-zinc-300"
                  >
                    {formatCellValue(row[columnName])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Card className="px-4 py-4">
      <p className="text-sm text-zinc-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-zinc-100">{value}</p>
    </Card>
  );
}

function IconDeleteButton({
  ariaLabel,
  disabled,
  onClick,
}: {
  ariaLabel: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <Button
      aria-label={ariaLabel}
      title={ariaLabel}
      onClick={onClick}
      disabled={disabled}
      variant="destructive"
      size="icon"
    >
      <X className="h-4 w-4" />
    </Button>
  );
}

function formatCellValue(value: unknown) {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
  }).format(new Date(value));
}

function formatMetricName(value: string) {
  return value
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatMetricValue(value: number) {
  return new Intl.NumberFormat("en", {
    maximumFractionDigits: 4,
  }).format(value);
}
