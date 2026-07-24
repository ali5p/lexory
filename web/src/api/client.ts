import type {
  ExerciseAnswerRequest,
  ExerciseAnswerResponse,
  SubmitRequest,
  SubmitResponse,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(detail || `Request failed (${response.status})`);
  }

  return (await response.json()) as T;
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
