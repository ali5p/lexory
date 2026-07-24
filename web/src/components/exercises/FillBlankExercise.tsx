import { useState } from "react";
import type { ExerciseAnswerResponse, FillBlankPayload } from "../../api/types";
import { submitExerciseAnswer } from "../../api/client";

type Props = {
  exerciseId: string;
  userId: string;
  payload: FillBlankPayload;
};

export function FillBlankExercise({ exerciseId, userId, payload }: Props) {
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<ExerciseAnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const parts = payload.sentence.split("___");

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!answer.trim() || result) {
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await submitExerciseAnswer(exerciseId, {
        user_id: userId,
        answer: answer.trim(),
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
      <p className="exercise-type">Fill in the blank</p>
      <p className="exercise-sentence">
        {parts[0]}
        <input
          type="text"
          value={answer}
          disabled={Boolean(result) || loading}
          onChange={(event) => setAnswer(event.target.value)}
          aria-label="Your answer"
          autoComplete="off"
        />
        {parts.slice(1).join("___")}
      </p>
      {payload.hint ? <p className="exercise-hint">Hint: {payload.hint}</p> : null}
      <button type="submit" disabled={!answer.trim() || Boolean(result) || loading}>
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
