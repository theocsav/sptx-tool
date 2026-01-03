export type Run = {
  id: number;
  run_name: string;
  status: string;
  stage?: string | null;
  run_dir?: string | null;
  output_dir?: string | null;
  config_path?: string | null;
  job_id?: string | null;
  message?: string | null;
  created_at: string;
  updated_at: string;
};

export type RunCreatePayload = {
  run_name: string;
  preset_path?: string;
  config?: Record<string, unknown>;
  submit?: boolean;
  queue?: boolean;
};

export type RunRerunPayload = {
  run_name: string;
  submit?: boolean;
  queue?: boolean;
};

export type Preset = {
  id: string;
  label?: string;
  organ?: string;
  platform?: string;
  path?: string;
  stages?: string[];
  run_name?: string;
  reference_h5ad_path?: string;
  output_dir?: string;
  ref_model_dir?: string;
  template_path?: string;
  post_nmf_notebook_path?: string;
  post_nmf_mode?: string;
  rcausal_notebook_path?: string;
  rcausal_script_path?: string;
  rcausal_mode?: string;
  rcausal_parameters?: Record<string, unknown>;
  rcausal_args?: string[];
  rcausal_output_dir?: string;
  rcausal_h5ad_path?: string;
  rcausal_niche_h5ad_path?: string;
  rcausal_neighborhood_h5ad_path?: string;
  mlp_script_path?: string;
  default_resources?: {
    time?: string;
    mem?: string;
    cpus_per_task?: number;
    qos?: string;
  };
  default_params?: {
    mode?: string;
    n_components?: number;
    k_min?: number;
    k_max?: number;
  };
  slurm?: {
    enabled?: boolean;
    job_name?: string;
    time?: string;
    mem?: string;
    cpus_per_task?: number;
    account?: string;
    partition?: string;
    qos?: string;
    mail_user?: string;
    mail_type?: string;
    conda_env?: string;
  };
  preflight_slurm?: Record<string, unknown>;
  version?: string;
};

export type Dataset = {
  id: string;
  label?: string;
  organ?: string;
  platform?: string;
  staged_path?: string;
  cell_metadata_path?: string;
  recommended_preset?: string;
  schema_manifest?: Record<string, unknown>;
  metadata_columns?: string[];
  notes?: string;
};

export type PreflightPayload = {
  preset_path?: string;
  config?: Record<string, unknown>;
  check_paths?: boolean;
};

export type PreflightResponse = {
  ok: boolean;
  errors: string[];
  warnings: string[];
  checks: Record<string, unknown>;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const CSRF_COOKIE_NAME = "sptx_csrf";
const CSRF_HEADER_NAME = "X-CSRF-Token";
let csrfToken: string | null = null;

function readCookie(name: string) {
  if (typeof document === "undefined") {
    return null;
  }
  const match = document.cookie.split("; ").find((item) => item.startsWith(`${name}=`));
  if (!match) {
    return null;
  }
  return decodeURIComponent(match.split("=")[1] || "");
}

function getCsrfToken() {
  return csrfToken || readCookie(CSRF_COOKIE_NAME);
}

function setCsrfToken(token?: string) {
  if (token) {
    csrfToken = token;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method || "GET").toUpperCase();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init?.headers || {}),
  };
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    const token = getCsrfToken();
    if (token) {
      headers[CSRF_HEADER_NAME] = token;
    }
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function fetchRuns() {
  return apiFetch<Run[]>("/runs");
}

export async function fetchRun(runId: number) {
  return apiFetch<Run>(`/runs/${runId}`);
}

export async function fetchPresets(params?: { organ?: string; platform?: string }) {
  const query = new URLSearchParams();
  if (params?.organ) {
    query.set("organ", params.organ);
  }
  if (params?.platform) {
    query.set("platform", params.platform);
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiFetch<Preset[]>(`/presets${suffix}`);
}

export async function fetchDatasets(params?: {
  organ?: string;
  platform?: string;
  preset_id?: string;
}) {
  const query = new URLSearchParams();
  if (params?.organ) {
    query.set("organ", params.organ);
  }
  if (params?.platform) {
    query.set("platform", params.platform);
  }
  if (params?.preset_id) {
    query.set("preset_id", params.preset_id);
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiFetch<Dataset[]>(`/datasets${suffix}`);
}

export async function createRun(payload: RunCreatePayload) {
  return apiFetch<Run>("/runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function rerunRun(runId: number, payload: RunRerunPayload) {
  return apiFetch<Run>(`/runs/${runId}/rerun`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function preflightRun(payload: PreflightPayload) {
  return apiFetch<PreflightResponse>("/runs/preflight", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchArtifacts(runId: number, path = "") {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return apiFetch<{ items: { path: string; size: string }[] }>(`/runs/${runId}/artifacts${query}`);
}

export async function fetchLogs(runId: number, path?: string) {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return apiFetch<{ path: string; content: string }>(`/runs/${runId}/logs${query}`);
}

export function artifactUrl(runId: number, path: string) {
  return `${API_BASE}/runs/${runId}/artifact?path=${encodeURIComponent(path)}`;
}

export async function cancelRun(runId: number) {
  return apiFetch<{ status: string }>(`/runs/${runId}/cancel`, { method: "POST" });
}

export async function login(username: string, password: string) {
  const data = await apiFetch<{ username: string; csrf_token?: string }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setCsrfToken(data.csrf_token);
  return data;
}

export async function logout() {
  const data = await apiFetch<{ status: string }>("/auth/logout", { method: "POST" });
  csrfToken = null;
  return data;
}

export async function fetchMe() {
  const data = await apiFetch<{ username: string; csrf_token?: string }>("/auth/me");
  setCsrfToken(data.csrf_token);
  return data;
}
