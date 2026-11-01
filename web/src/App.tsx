import { useEffect, useState } from "react";
import { logoutUser, refreshAccessToken, setAccessToken } from "./api/client";
import { AuthPanel } from "./components/AuthPanel";
import { GoogleOAuthCallback } from "./components/GoogleOAuthCallback";
import { SubmitPanel } from "./components/SubmitPanel";
import "./App.css";

function App() {
  const [checkingSession, setCheckingSession] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [googleCallback, setGoogleCallback] = useState(
    () => window.location.pathname === "/auth/callback",
  );

  useEffect(() => {
    if (window.location.pathname === "/auth/callback") {
      setCheckingSession(false);
      return;
    }

    let active = true;
    refreshAccessToken()
      .then((restored) => {
        if (active) {
          setAuthenticated(restored);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setAuthError(
            error instanceof Error ? error.message : "Could not restore session.",
          );
        }
      })
      .finally(() => {
        if (active) {
          setCheckingSession(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  async function handleLogout() {
    setAuthError(null);
    try {
      await logoutUser();
      setAuthenticated(false);
    } catch (error) {
      setAuthError(
        error instanceof Error ? error.message : "Could not sign out.",
      );
    }
  }

  function handleAuthenticated(accessToken: string) {
    setAccessToken(accessToken);
    setAuthenticated(true);
    setAuthError(null);
    setGoogleCallback(false);
  }

  function handleGoogleOAuthError(message: string) {
    setAuthError(message);
    setGoogleCallback(false);
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Lexory</h1>
        <p>Submit English text, get lessons and exercises.</p>
      </header>
      <main>
        {googleCallback ? (
          <GoogleOAuthCallback
            onAuthenticated={handleAuthenticated}
            onError={handleGoogleOAuthError}
          />
        ) : checkingSession ? (
          <p>Checking your session…</p>
        ) : authenticated ? (
          <>
            <div className="account-bar">
              <span>Signed in</span>
              <button type="button" onClick={handleLogout}>
                Sign out
              </button>
            </div>
            {authError ? <p className="feedback error">{authError}</p> : null}
            <SubmitPanel />
          </>
        ) : (
          <>
            {authError ? <p className="feedback error">{authError}</p> : null}
            <AuthPanel onAuthenticated={handleAuthenticated} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
