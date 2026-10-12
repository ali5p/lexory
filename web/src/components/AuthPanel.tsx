import { useState, type FormEvent } from "react";
import { loginUser, registerUser } from "../api/client";

type Props = {
  onAuthenticated: (accessToken: string) => void;
};

export function AuthPanel({ onAuthenticated }: Props) {
  const [isRegistering, setIsRegistering] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      if (isRegistering) {
        await registerUser(email, password);
      }
      const token = await loginUser(email, password);
      onAuthenticated(token.access_token);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setLoading(false);
    }
  }

  function toggleMode() {
    setIsRegistering((registering) => !registering);
    setError(null);
  }

  return (
    <section className="auth-panel">
      <h2>{isRegistering ? "Create an account" : "Sign in"}</h2>
      <form className="auth-form" onSubmit={handleSubmit}>
        <label className="field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="email"
            required
          />
        </label>
        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete={isRegistering ? "new-password" : "current-password"}
            minLength={8}
            maxLength={128}
            required
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading
            ? "Please wait…"
            : isRegistering
              ? "Create account"
              : "Sign in"}
        </button>
        {error ? <p className="feedback error">{error}</p> : null}
      </form>
      <button
        className="text-button"
        type="button"
        onClick={toggleMode}
        disabled={loading}
      >
        {isRegistering ? "Already have an account? Sign in" : "Create an account"}
      </button>
    </section>
  );
}
