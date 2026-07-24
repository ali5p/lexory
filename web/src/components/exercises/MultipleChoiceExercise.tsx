import { useState } from "react";
import type { ExerciseAnswerResponse, MultipleChoicePayload } from "../../api/types";
import { submitExerciseAnswer } from "../../api/client";

type Props = {
  exerciseId: string;
  userId: string;
  payload: MultipleChoicePayload;
};

export function MultipleChoiceExercise({ exerciseId, userId, payload }: Props) {
  const [selected, setSelected] = useState("");
  const [result, setResult] = useState<ExerciseAnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const parts = payload.sentence.split("___");

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!selected || result) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await submitExerciseAnswer(exerciseId, {
        user_id: userId,
        selected_option: selected,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not submit answer.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="exercise" onSubmit={handleSubmit}>
      <p className="exercise-type">Multiple choice</p>
      <p className="exercise-sentence">
        {parts.map((part, index) => (
          <span key={`${index}-${part}`}>
            {part}
            {index < parts.length - 1 ? (
              <select
                value={selected}
                disabled={Boolean(result) || loading}
                onChange={(event) => setSelected(event.target.value)}
                aria-label="Choose the correct option"
              >
                <option value="">—</option>
                {payload.options.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            ) : null}
          </span>
        ))}
      </p>
      <button type="submit" disabled={!selected || Boolean(result) || loading}>
        {loading ? "Checking…" : "Check answer"}
      </button>
      {error ? <p className="feedback error">{error}</p> : null}
      {result ? (
        <p className={`feedback ${result.correct ? "success" : "error"}`}>
          {result.explanation}
        </p>
      ) : null}
    </form>
  );
}
