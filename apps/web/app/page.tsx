"use client";

import {
  FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

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

type FeatureImportance = {
  feature: string;
  importance: number;
};

type ModelTrainingResult = {
  problem_type: string;
  model_type: string;
  metrics: Record<string, number>;
  feature_importance: FeatureImportance[];
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
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isLoadingProjects, setIsLoadingProjects] = useState(true);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [isUploadingCsv, setIsUploadingCsv] = useState(false);
  const [isSavingDataset, setIsSavingDataset] = useState(false);
  const [isTrainingModel, setIsTrainingModel] = useState(false);
  const [isSendingChatMessage, setIsSendingChatMessage] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const activeProject = useMemo(
    () => projects.find((project) => project.id === activeProjectId) ?? null,
    [activeProjectId, projects],
  );

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
    setChatMessages([]);
    setChatInput("");
    setSuccessMessage(null);

    if (activeProjectId === null) {
      setDatasets([]);
      return;
    }

    loadDatasets(activeProjectId);
    loadChatHistory(activeProjectId);
  }, [activeProjectId, loadChatHistory, loadDatasets]);

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

  async function handleUploadPreview(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

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

      const response = await fetch(`${API_BASE_URL}/datasets/upload-preview`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Could not preview CSV.");
      }

      const preview = (await response.json()) as DatasetUploadPreview;
      setUploadPreview(preview);
      setTargetColumn(preview.column_names[0] ?? "");
      setModelTrainingResult(null);
    } catch (error) {
      setUploadPreview(null);
      setTargetColumn("");
      setModelTrainingResult(null);
      setErrorMessage(
        error instanceof Error ? error.message : "Could not preview CSV.",
      );
    } finally {
      setIsUploadingCsv(false);
    }
  }

  async function handleSaveDataset() {
    if (!activeProjectId || !uploadPreview) {
      return;
    }

    try {
      setIsSavingDataset(true);
      setErrorMessage(null);
      setSuccessMessage(null);

      const response = await fetch(
        `${API_BASE_URL}/projects/${activeProjectId}/datasets`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            filename: uploadPreview.filename,
            row_count: uploadPreview.rows,
            column_count: uploadPreview.columns,
            data: uploadPreview.data,
          }),
        },
      );

      if (!response.ok) {
        throw new Error("Could not save dataset.");
      }

      await loadDatasets(activeProjectId);
      setSuccessMessage("Dataset saved to project.");
      setUploadPreview(null);
      setSelectedCsvFile(null);
      setTargetColumn("");
      setModelTrainingResult(null);
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not save dataset.",
      );
    } finally {
      setIsSavingDataset(false);
    }
  }

  async function handleTrainModel() {
    if (!uploadPreview || !targetColumn) {
      setErrorMessage("Choose a target column before training.");
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
          data: uploadPreview.data,
          target_column: targetColumn,
        }),
      });

      if (!response.ok) {
        const errorBody = (await response.json().catch(() => null)) as {
          detail?: string;
        } | null;
        throw new Error(errorBody?.detail || "Could not train model.");
      }

      setModelTrainingResult((await response.json()) as ModelTrainingResult);
      setSuccessMessage("Baseline model trained.");
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
      const savedUserMessage = await saveProjectChatMessage(
        activeProjectId,
        "user",
        trimmedMessage,
      );
      setChatMessages((currentMessages) => [
        ...currentMessages,
        savedUserMessage,
      ]);

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: trimmedMessage,
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
      const savedAssistantMessage = await saveProjectChatMessage(
        activeProjectId,
        "assistant",
        assistantReply,
      );
      setChatMessages((currentMessages) => [
        ...currentMessages,
        savedAssistantMessage,
      ]);
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
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <div className="flex min-h-screen flex-col lg:flex-row">
        <aside className="border-b border-slate-200 bg-white lg:w-80 lg:border-b-0 lg:border-r">
          <div className="border-b border-slate-200 px-6 py-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-700">
              Workspace
            </p>
            <h1 className="mt-2 text-2xl font-semibold">
              AI Scientist Copilot
            </h1>
          </div>

          <form
            onSubmit={handleCreateProject}
            className="space-y-3 border-b border-slate-200 px-6 py-5"
          >
            <div>
              <label
                htmlFor="project-name"
                className="text-sm font-medium text-slate-700"
              >
                Project name
              </label>
              <input
                id="project-name"
                value={projectName}
                onChange={(event) => setProjectName(event.target.value)}
                className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                placeholder="Battery electrolyte study"
              />
            </div>

            <div>
              <label
                htmlFor="project-description"
                className="text-sm font-medium text-slate-700"
              >
                Description
              </label>
              <textarea
                id="project-description"
                value={projectDescription}
                onChange={(event) =>
                  setProjectDescription(event.target.value)
                }
                className="mt-1 min-h-20 w-full resize-none rounded-md border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                placeholder="Optional project context"
              />
            </div>

            <button
              type="submit"
              disabled={isCreatingProject}
              className="w-full rounded-md bg-slate-950 px-3 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {isCreatingProject ? "Creating..." : "Create Project"}
            </button>
          </form>

          <section className="px-3 py-4">
            <div className="mb-2 flex items-center justify-between px-3">
              <h2 className="text-sm font-semibold text-slate-800">
                Projects
              </h2>
              <span className="text-xs text-slate-500">{projects.length}</span>
            </div>

            {isLoadingProjects ? (
              <p className="px-3 py-4 text-sm text-slate-500">
                Loading projects...
              </p>
            ) : projects.length === 0 ? (
              <p className="px-3 py-4 text-sm leading-6 text-slate-500">
                Create a project to start building a scientific workspace.
              </p>
            ) : (
              <div className="space-y-1">
                {projects.map((project) => {
                  const isActive = project.id === activeProjectId;
                  return (
                    <button
                      key={project.id}
                      type="button"
                      onClick={() => setActiveProjectId(project.id)}
                      className={`w-full rounded-md px-3 py-3 text-left transition ${
                        isActive
                          ? "bg-cyan-50 text-cyan-950 ring-1 ring-cyan-200"
                          : "text-slate-700 hover:bg-slate-100"
                      }`}
                    >
                      <span className="block truncate text-sm font-semibold">
                        {project.name}
                      </span>
                      <span className="mt-1 block truncate text-xs text-slate-500">
                        {project.description || "No description"}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </section>
        </aside>

        <section className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
          {errorMessage ? (
            <div className="mb-5 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {errorMessage}
            </div>
          ) : null}

          {successMessage ? (
            <div className="mb-5 rounded-md border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
              {successMessage}
            </div>
          ) : null}

          {activeProject ? (
            <div className="space-y-6">
              <header className="border-b border-slate-200 pb-5">
                <p className="text-sm font-medium text-slate-500">
                  Active project
                </p>
                <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                  {activeProject.name}
                </h2>
                {activeProject.description ? (
                  <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
                    {activeProject.description}
                  </p>
                ) : null}
              </header>

              <div className="grid gap-4 sm:grid-cols-3">
                <Metric label="Datasets" value={datasets.length.toString()} />
                <Metric
                  label="Rows saved"
                  value={datasets
                    .reduce((total, dataset) => total + dataset.row_count, 0)
                    .toLocaleString()}
                />
                <Metric
                  label="Columns tracked"
                  value={datasets
                    .reduce(
                      (total, dataset) => total + dataset.column_count,
                      0,
                    )
                    .toLocaleString()}
                />
              </div>

              <CsvUploadPanel
                isSavingDataset={isSavingDataset}
                isTrainingModel={isTrainingModel}
                isUploadingCsv={isUploadingCsv}
                modelTrainingResult={modelTrainingResult}
                onFileChange={(file) => {
                  setSelectedCsvFile(file);
                  setUploadPreview(null);
                  setTargetColumn("");
                  setModelTrainingResult(null);
                  setSuccessMessage(null);
                }}
                onSaveDataset={handleSaveDataset}
                onTargetColumnChange={(columnName) => {
                  setTargetColumn(columnName);
                  setModelTrainingResult(null);
                }}
                onTrainModel={handleTrainModel}
                onUploadPreview={handleUploadPreview}
                selectedCsvFile={selectedCsvFile}
                targetColumn={targetColumn}
                uploadPreview={uploadPreview}
              />

              <section>
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="text-lg font-semibold">Saved datasets</h3>
                  {isLoadingDatasets ? (
                    <span className="text-sm text-slate-500">Loading...</span>
                  ) : null}
                </div>

                {datasets.length === 0 && !isLoadingDatasets ? (
                  <div className="rounded-md border border-dashed border-slate-300 bg-white px-5 py-10 text-center">
                    <h4 className="text-base font-semibold text-slate-800">
                      No datasets saved yet
                    </h4>
                    <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">
                      Upload a CSV, inspect the preview, and save it into this
                      project.
                    </p>
                  </div>
                ) : (
                  <div className="overflow-hidden rounded-md border border-slate-200 bg-white">
                    <div className="grid grid-cols-12 border-b border-slate-200 bg-slate-100 px-4 py-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
                      <span className="col-span-6">Filename</span>
                      <span className="col-span-2 text-right">Rows</span>
                      <span className="col-span-2 text-right">Columns</span>
                      <span className="col-span-2 text-right">Saved</span>
                    </div>

                    {datasets.map((dataset) => (
                      <div
                        key={dataset.id}
                        className="grid grid-cols-12 items-center border-b border-slate-100 px-4 py-4 text-sm last:border-b-0"
                      >
                        <span className="col-span-6 truncate font-medium text-slate-900">
                          {dataset.filename}
                        </span>
                        <span className="col-span-2 text-right text-slate-600">
                          {dataset.row_count.toLocaleString()}
                        </span>
                        <span className="col-span-2 text-right text-slate-600">
                          {dataset.column_count.toLocaleString()}
                        </span>
                        <span className="col-span-2 text-right text-slate-500">
                          {formatDate(dataset.created_at)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>
          ) : (
            <div className="flex min-h-[60vh] items-center justify-center">
              <div className="max-w-lg text-center">
                <p className="text-sm font-semibold uppercase tracking-wide text-cyan-700">
                  AI Scientist Copilot
                </p>
                <h2 className="mt-3 text-3xl font-semibold tracking-tight">
                  Create or select a project
                </h2>
                <p className="mt-3 text-sm leading-6 text-slate-600">
                  Projects organize datasets, analysis, models, and future
                  copilot memory for each scientific workspace.
                </p>
              </div>
            </div>
          )}
        </section>

        <CopilotChatPanel
          activeProjectId={activeProjectId}
          chatInput={chatInput}
          datasetSummary={datasetChatSummary}
          isSending={isSendingChatMessage}
          messages={chatMessages}
          modelResult={modelTrainingResult}
          onInputChange={setChatInput}
          onSend={handleSendChatMessage}
        />
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
  onInputChange,
  onSend,
}: {
  activeProjectId: number | null;
  chatInput: string;
  datasetSummary: DatasetChatSummary | null;
  isSending: boolean;
  messages: ChatMessage[];
  modelResult: ModelTrainingResult | null;
  onInputChange: (value: string) => void;
  onSend: (event: FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <aside className="border-t border-slate-200 bg-white lg:flex lg:w-96 lg:flex-col lg:border-l lg:border-t-0">
      <div className="border-b border-slate-200 px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-700">
              Copilot
            </p>
            <h2 className="mt-1 text-lg font-semibold">Scientist chat</h2>
          </div>
          <div className="flex gap-1.5 pt-1">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                datasetSummary ? "bg-emerald-500" : "bg-slate-300"
              }`}
              title={
                datasetSummary
                  ? "Dataset preview is included"
                  : "No dataset preview available"
              }
            />
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                modelResult ? "bg-cyan-600" : "bg-slate-300"
              }`}
              title={
                modelResult
                  ? "Model result is included"
                  : "No model result available"
              }
            />
          </div>
        </div>
      </div>

      <div className="flex max-h-[28rem] flex-1 flex-col gap-3 overflow-y-auto px-5 py-4 lg:max-h-none">
        {messages.length === 0 ? (
          <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-sm leading-6 text-slate-600">
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
                className={`max-w-[85%] rounded-md px-3 py-2 text-sm leading-6 ${
                  message.role === "user"
                    ? "bg-slate-950 text-white"
                    : "border border-slate-200 bg-slate-50 text-slate-700"
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
            <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-500">
              Thinking...
            </div>
          </div>
        ) : null}
      </div>

      <form onSubmit={onSend} className="border-t border-slate-200 p-4">
        <label htmlFor="copilot-message" className="sr-only">
          Copilot message
        </label>
        <textarea
          id="copilot-message"
          value={chatInput}
          onChange={(event) => onInputChange(event.target.value)}
          className="min-h-24 w-full resize-none rounded-md border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
          placeholder="Ask the copilot about this workspace..."
        />
        <button
          type="submit"
          disabled={!activeProjectId || !chatInput.trim() || isSending}
          className="mt-3 w-full rounded-md bg-cyan-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:bg-cyan-300"
        >
          {isSending ? "Sending..." : "Send"}
        </button>
      </form>
    </aside>
  );
}

function CsvUploadPanel({
  isSavingDataset,
  isTrainingModel,
  isUploadingCsv,
  modelTrainingResult,
  onFileChange,
  onSaveDataset,
  onTargetColumnChange,
  onTrainModel,
  onUploadPreview,
  selectedCsvFile,
  targetColumn,
  uploadPreview,
}: {
  isSavingDataset: boolean;
  isTrainingModel: boolean;
  isUploadingCsv: boolean;
  modelTrainingResult: ModelTrainingResult | null;
  onFileChange: (file: File | null) => void;
  onSaveDataset: () => void;
  onTargetColumnChange: (columnName: string) => void;
  onTrainModel: () => void;
  onUploadPreview: (event: FormEvent<HTMLFormElement>) => void;
  selectedCsvFile: File | null;
  targetColumn: string;
  uploadPreview: DatasetUploadPreview | null;
}) {
  return (
    <section className="rounded-md border border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-5 py-4">
        <h3 className="text-lg font-semibold">CSV upload</h3>
      </div>

      <form
        onSubmit={onUploadPreview}
        className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center"
      >
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
          className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-200"
        />
        <button
          type="submit"
          disabled={!selectedCsvFile || isUploadingCsv}
          className="rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
        >
          {isUploadingCsv ? "Previewing..." : "Upload Preview"}
        </button>
      </form>

      {uploadPreview ? (
        <div className="border-t border-slate-200 px-5 py-5">
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <h4 className="text-base font-semibold">
                {uploadPreview.filename}
              </h4>
              <p className="mt-1 text-sm text-slate-500">
                {uploadPreview.rows.toLocaleString()} rows,{" "}
                {uploadPreview.columns.toLocaleString()} columns
              </p>
            </div>
            <button
              type="button"
              onClick={onSaveDataset}
              disabled={isSavingDataset}
              className="rounded-md bg-cyan-700 px-4 py-2 text-sm font-semibold text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:bg-cyan-300"
            >
              {isSavingDataset ? "Saving..." : "Save Dataset to Project"}
            </button>
          </div>

          <div className="mb-4 flex flex-wrap gap-2">
            {uploadPreview.column_names.map((columnName) => (
              <span
                key={columnName}
                className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700"
              >
                {columnName}
              </span>
            ))}
          </div>

          <div className="mb-4 rounded-md border border-slate-200 bg-slate-50 px-4 py-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="flex-1">
                <label
                  htmlFor="target-column"
                  className="text-sm font-medium text-slate-700"
                >
                  Target column
                </label>
                <select
                  id="target-column"
                  value={targetColumn}
                  onChange={(event) =>
                    onTargetColumnChange(event.target.value)
                  }
                  className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm outline-none transition focus:border-cyan-600 focus:ring-2 focus:ring-cyan-100"
                >
                  {uploadPreview.column_names.map((columnName) => (
                    <option key={columnName} value={columnName}>
                      {columnName}
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="button"
                onClick={onTrainModel}
                disabled={!targetColumn || isTrainingModel}
                className="rounded-md bg-slate-950 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
              >
                {isTrainingModel ? "Training..." : "Train Model"}
              </button>
            </div>
          </div>

          {modelTrainingResult ? (
            <ModelTrainingResults result={modelTrainingResult} />
          ) : null}

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
    </section>
  );
}

function ModelTrainingResults({ result }: { result: ModelTrainingResult }) {
  const topFeatureImportance = result.feature_importance.slice(0, 10);

  return (
    <div className="mb-4 grid gap-4 lg:grid-cols-2">
      <div className="rounded-md border border-slate-200 bg-white px-4 py-4">
        <h4 className="text-sm font-semibold text-slate-900">
          Model summary
        </h4>
        <dl className="mt-4 grid gap-3 text-sm">
          <div className="flex items-center justify-between gap-4">
            <dt className="text-slate-500">Problem type</dt>
            <dd className="font-medium capitalize text-slate-900">
              {result.problem_type}
            </dd>
          </div>
          <div className="flex items-center justify-between gap-4">
            <dt className="text-slate-500">Model type</dt>
            <dd className="font-medium text-slate-900">
              {result.model_type}
            </dd>
          </div>
        </dl>
      </div>

      <div className="rounded-md border border-slate-200 bg-white px-4 py-4">
        <h4 className="text-sm font-semibold text-slate-900">Metrics</h4>
        <dl className="mt-4 grid gap-3 text-sm">
          {Object.entries(result.metrics).map(([metricName, metricValue]) => (
            <div
              key={metricName}
              className="flex items-center justify-between gap-4"
            >
              <dt className="text-slate-500">{formatMetricName(metricName)}</dt>
              <dd className="font-medium tabular-nums text-slate-900">
                {formatMetricValue(metricValue)}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="overflow-hidden rounded-md border border-slate-200 bg-white lg:col-span-2">
        <div className="border-b border-slate-200 bg-slate-100 px-4 py-3 text-sm font-semibold">
          Top feature importance
        </div>
        {topFeatureImportance.length === 0 ? (
          <p className="px-4 py-4 text-sm text-slate-500">
            No feature importance values returned.
          </p>
        ) : (
          <div className="max-h-80 overflow-auto">
            <table className="w-full min-w-[32rem] text-left text-sm">
              <thead className="bg-white text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="px-4 py-3">Feature</th>
                  <th className="px-4 py-3 text-right">Importance</th>
                </tr>
              </thead>
              <tbody>
                {topFeatureImportance.map((item) => (
                  <tr key={item.feature} className="border-t border-slate-100">
                    <td className="max-w-96 truncate px-4 py-3 font-medium text-slate-900">
                      {item.feature}
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums text-slate-600">
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
    <div className="overflow-hidden rounded-md border border-slate-200">
      <div className="border-b border-slate-200 bg-slate-100 px-4 py-3 text-sm font-semibold">
        Profile
      </div>
      <div className="max-h-80 overflow-auto">
        <table className="w-full min-w-96 text-left text-sm">
          <thead className="bg-white text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-4 py-3">Column</th>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3 text-right">Missing</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(columnTypes).map(([columnName, columnType]) => (
              <tr key={columnName} className="border-t border-slate-100">
                <td className="px-4 py-3 font-medium text-slate-900">
                  {columnName}
                </td>
                <td className="px-4 py-3 text-slate-600">{columnType}</td>
                <td className="px-4 py-3 text-right text-slate-600">
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
    <div className="overflow-hidden rounded-md border border-slate-200">
      <div className="border-b border-slate-200 bg-slate-100 px-4 py-3 text-sm font-semibold">
        Preview
      </div>
      <div className="max-h-80 overflow-auto">
        <table className="w-full min-w-[36rem] text-left text-sm">
          <thead className="bg-white text-xs uppercase tracking-wide text-slate-500">
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
              <tr key={rowIndex} className="border-t border-slate-100">
                {columnNames.map((columnName) => (
                  <td
                    key={columnName}
                    className="max-w-44 truncate px-4 py-3 text-slate-700"
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
    <div className="rounded-md border border-slate-200 bg-white px-4 py-4">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-950">{value}</p>
    </div>
  );
}

async function saveProjectChatMessage(
  projectId: number,
  role: ChatMessage["role"],
  content: string,
) {
  const response = await fetch(`${API_BASE_URL}/projects/${projectId}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ role, content }),
  });

  if (!response.ok) {
    throw new Error("Could not save chat message.");
  }

  return (await response.json()) as ChatMessage;
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
