import type {
  ExerciseAnswerRequest,
  ExerciseAnswerResponse,
  SubmitRequest,
  SubmitResponse,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

type TokenResponse = {
  access_token: string;
  token_type: "bearer";
};

type UserResponse = {
  id: string;
  email: string;
};

class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

let accessToken: string | null = null;
let refreshInFlight: Promise<boolean> | null = null;

function isAuthRequest(path: string): boolean {
  return path.startsWith("/auth/");
}

async function fetchApi(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (accessToken && !isAuthRequest(path)) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  return fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers,
  });
}

async function responseError(response: Response): Promise<ApiError> {
  let detail = response.statusText;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      detail = body.detail;
    } else if (body.detail !== undefined) {
      detail = JSON.stringify(body.detail);
    }
  } catch {
    // Keep the HTTP status text if the error response isn't JSON.
  }
  return new ApiError(detail || `Request failed (${response.status})`, response.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response = await fetchApi(path, init);
  if (response.status === 401 && accessToken && !isAuthRequest(path)) {
    if (await refreshAccessToken()) {
      response = await fetchApi(path, init);
    }
  }
  if (!response.ok) {
    throw await responseError(response);
  }
  return (await response.json()) as T;
}

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

export async function refreshAccessToken(): Promise<boolean> {
  if (refreshInFlight) {
    return refreshInFlight;
  }

  refreshInFlight = (async () => {
    const response = await fetchApi("/auth/refresh", {
      method: "POST",
      body: "{}",
    });
    if (response.status === 401) {
      accessToken = null;
      return false;
    }
    if (!response.ok) {
      throw await responseError(response);
    }
    const body = (await response.json()) as TokenResponse;
    accessToken = body.access_token;
    return true;
  })();

  try {
    return await refreshInFlight;
  } finally {
    refreshInFlight = null;
  }
}

export function registerUser(
  email: string,
  password: string,
): Promise<UserResponse> {
  return request<UserResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export function loginUser(
  email: string,
  password: string,
): Promise<TokenResponse> {
  return request<TokenResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function logoutUser(): Promise<void> {
  await request<{ detail: string }>("/auth/logout", {
    method: "POST",
    body: "{}",
  });
  accessToken = null;
}

export function submitText(payload: SubmitRequest): Promise<SubmitResponse> {
  return request<SubmitResponse>("/submit", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function submitExerciseAnswer(
  exerciseId: string,
  payload: ExerciseAnswerRequest,
): Promise<ExerciseAnswerResponse> {
  return request<ExerciseAnswerResponse>(`/exercises/${exerciseId}/answer`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
