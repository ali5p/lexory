import { useState } from "react";
import type { SubmitResponse } from "../api/types";
import { submitText } from "../api/client";
import { LessonItemCard } from "./LessonItemCard";

export function SubmitPanel() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SubmitResponse | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!text.trim()) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await submitText({ text: text.trim() });
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
          <span>Your text</span>
          <textarea
            rows={5}
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Write a few sentences in English…"
          />
        </label>
        <button type="submit" disabled={loading || !text.trim()}>
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
              />
            ))
          )}
        </section>
      ) : null}
    </div>
  );
}
