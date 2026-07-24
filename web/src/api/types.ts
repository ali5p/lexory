export type MultipleChoicePayload = {
  type: "multiple_choice";
  instruction: string;
  sentence: string;
  options: string[];
};

export type FillBlankPayload = {
  type: "fill_blank";
  instruction: string;
  sentence: string;
  hint: string | null;
};

export type ExercisePayloadBody = MultipleChoicePayload | FillBlankPayload;

export type Exercise = {
  exercise_id: string;
  mistake_type: string;
  source_sentence: string;
  payload: ExercisePayloadBody;
};

export type LessonTarget = {
  mistake_id: string;
  rule_id: string;
  mistake_type: string;
  text: string;
  rule_message: string;
};

export type LessonContent = {
  topic: string;
  explanation: string;
  exercises: Exercise[];
};

export type LessonItem = {
  lesson_artifact_id: string | null;
  target: LessonTarget | null;
  lesson: LessonContent;
};

export type DetectedMistake = {
  mistake_id: string;
  rule_id: string;
  mistake_type: string;
  text: string;
  rule_message: string;
  selected_for_lesson: boolean;
};

export type SubmitRequest = {
  text: string;
  user_id: string;
};

export type SubmitResponse = {
  user_text_id: string | null;
  session_id: string;
  detected_mistakes: DetectedMistake[];
  lesson_items: LessonItem[];
};

export type ExerciseAnswerRequest = {
  user_id: string;
  selected_option?: string;
  answer?: string;
};

export type ExerciseAnswerResponse = {
  correct: boolean;
  explanation: string;
  exercise_attempt_id: string;
};
