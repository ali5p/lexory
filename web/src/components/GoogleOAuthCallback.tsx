import { useEffect, useRef } from "react";
import { completeGoogleSignIn } from "../api/googleOAuth";

type Props = {
  onAuthenticated: (accessToken: string) => void;
  onError: (message: string) => void;
};

export function GoogleOAuthCallback({ onAuthenticated, onError }: Props) {
  const started = useRef(false);

  useEffect(() => {
    if (started.current) {
      return;
    }
    started.current = true;

    void completeGoogleSignIn()
      .then(({ access_token: accessToken }) => {
        onAuthenticated(accessToken);
      })
      .catch((error: unknown) => {
        onError(
          error instanceof Error ? error.message : "Google sign-in failed.",
        );
      });
  }, [onAuthenticated, onError]);

  return <p>Completing Google sign-in…</p>;
}
