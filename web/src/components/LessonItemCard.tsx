import type { LessonItem } from "../api/types";
import { ExerciseRenderer } from "./exercises/ExerciseRenderer";

type Props = {
  item: LessonItem;
  userId: string;
};

export function LessonItemCard({ item, userId }: Props) {
  const { lesson, target } = item;

  return (
    <article className="lesson-card">
      <header>
        <h2>{lesson.topic}</h2>
        {target ? (
          <p className="lesson-target">
            <span className="badge">{target.mistake_type}</span>
            {target.rule_message || target.text}
          </p>
        ) : null}
      </header>
      <p className="lesson-explanation">{lesson.explanation}</p>
      {target?.text ? (
        <blockquote className="source-sentence">{target.text}</blockquote>
      ) : null}
      {lesson.exercises.length === 0 ? (
        <p className="no-exercise">No exercise for this lesson yet.</p>
      ) : (
        lesson.exercises.map((exercise) => (
          <ExerciseRenderer key={exercise.exercise_id} exercise={exercise} userId={userId} />
        ))
      )}
    </article>
  );
}
