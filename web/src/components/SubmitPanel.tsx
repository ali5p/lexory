import { useState } from "react";
import type { SubmitResponse } from "../api/types";
import { submitText } from "../api/client";
import { LessonItemCard } from "./LessonItemCard";

const USER_ID_KEY = "lexory_user_id";

function loadUserId(): string {
  const stored = localStorage.getItem(USER_ID_KEY);
  if (stored) {
    return stored;
  }
  const generated =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `user-${Date.now()}`;
  localStorage.setItem(USER_ID_KEY, generated);
  return generated;
}

export function SubmitPanel() {
  const [userId, setUserId] = useState(loadUserId);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SubmitResponse | null>(null);

  function handleUserIdChange(value: string) {
    setUserId(value);
    localStorage.setItem(USER_ID_KEY, value);
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!text.trim()) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await submitText({ text: text.trim(), user_id: userId.trim() });
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Submit failed.");
      setResponse(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="submit-panel">
      <form className="submit-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>User ID</span>
          <input
            type="text"
            value={userId}
            onChange={(event) => handleUserIdChange(event.target.value)}
            autoComplete="off"
          />
        </label>
        <label className="field">
          <span>Your text</span>
          <textarea
            rows={5}
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Write a few sentences in English…"
          />
        </label>
        <button type="submit" disabled={loading || !text.trim() || !userId.trim()}>
          {loading ? "Analyzing…" : "Submit"}
        </button>
        {error ? <p className="feedback error">{error}</p> : null}
      </form>

      {response ? (
        <section className="results">
          <p className="session-meta">Session: {response.session_id}</p>
          {response.lesson_items.length === 0 ? (
            <p className="no-lessons">No lessons generated for this submission.</p>
          ) : (
            response.lesson_items.map((item, index) => (
              <LessonItemCard
                key={item.lesson_artifact_id ?? `lesson-${index}`}
                item={item}
                userId={userId}
              />
            ))
          )}
        </section>
      ) : null}
    </div>
  );
}
