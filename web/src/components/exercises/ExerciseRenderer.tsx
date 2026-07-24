import type { Exercise } from "../../api/types";
import { FillBlankExercise } from "./FillBlankExercise";
import { MultipleChoiceExercise } from "./MultipleChoiceExercise";

type Props = {
  exercise: Exercise;
  userId: string;
};

export function ExerciseRenderer({ exercise, userId }: Props) {
  if (exercise.payload.type === "multiple_choice") {
    return (
      <MultipleChoiceExercise
        exerciseId={exercise.exercise_id}
        userId={userId}
        payload={exercise.payload}
      />
    );
  }

  return (
    <FillBlankExercise
      exerciseId={exercise.exercise_id}
      userId={userId}
      payload={exercise.payload}
    />
  );
}
