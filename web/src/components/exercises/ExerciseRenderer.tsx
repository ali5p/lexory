import type { Exercise } from "../../api/types";
import { FillBlankExercise } from "./FillBlankExercise";
import { MultipleChoiceExercise } from "./MultipleChoiceExercise";

type Props = {
  exercise: Exercise;
};

export function ExerciseRenderer({ exercise }: Props) {
  if (exercise.payload.type === "multiple_choice") {
    return (
      <MultipleChoiceExercise
        exerciseId={exercise.exercise_id}
        payload={exercise.payload}
      />
    );
  }

  return (
    <FillBlankExercise
      exerciseId={exercise.exercise_id}
      payload={exercise.payload}
    />
  );
}
