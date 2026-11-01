import {
  exchangeGoogleCode,
  getGoogleAuthConfig,
  type TokenResponse,
} from "./client";

const GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth";
const OAUTH_STORAGE_KEY = "lexory.google.oauth";
const OAUTH_STATE_MAX_AGE_MS = 10 * 60 * 1000;

type OAuthTransaction = {
  state: string;
  nonce: string;
  codeVerifier: string;
  createdAt: number;
};

function randomUrlSafeString(byteLength: number): string {
  const bytes = crypto.getRandomValues(new Uint8Array(byteLength));
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replace(/=+$/, "");
}

async function createCodeChallenge(codeVerifier: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(codeVerifier),
  );
  let binary = "";
  for (const byte of new Uint8Array(digest)) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replace(/=+$/, "");
}

export async function startGoogleSignIn(): Promise<void> {
  const config = await getGoogleAuthConfig();
  const transaction: OAuthTransaction = {
    state: randomUrlSafeString(32),
    nonce: randomUrlSafeString(32),
    codeVerifier: randomUrlSafeString(32),
    createdAt: Date.now(),
  };
  sessionStorage.setItem(OAUTH_STORAGE_KEY, JSON.stringify(transaction));

  const authorizationUrl = new URL(GOOGLE_AUTH_ENDPOINT);
  authorizationUrl.search = new URLSearchParams({
    client_id: config.client_id,
    redirect_uri: config.redirect_uri,
    response_type: "code",
    scope: "openid email profile",
    state: transaction.state,
    nonce: transaction.nonce,
    code_challenge: await createCodeChallenge(transaction.codeVerifier),
    code_challenge_method: "S256",
  }).toString();

  window.location.assign(authorizationUrl);
}

function parseOAuthTransaction(
  serialized: string | null,
): OAuthTransaction | null {
  if (!serialized) {
    return null;
  }

  try {
    const value: unknown = JSON.parse(serialized);
    if (
      typeof value === "object" &&
      value !== null &&
      "state" in value &&
      "nonce" in value &&
      "codeVerifier" in value &&
      "createdAt" in value &&
      typeof value.state === "string" &&
      typeof value.nonce === "string" &&
      typeof value.codeVerifier === "string" &&
      typeof value.createdAt === "number"
    ) {
      return {
        state: value.state,
        nonce: value.nonce,
        codeVerifier: value.codeVerifier,
        createdAt: value.createdAt,
      };
    }
  } catch {
    return null;
  }
  return null;
}

export async function completeGoogleSignIn(): Promise<TokenResponse> {
  const callbackUrl = new URL(window.location.href);
  const code = callbackUrl.searchParams.get("code");
  const returnedState = callbackUrl.searchParams.get("state");
  const providerError = callbackUrl.searchParams.get("error");
  window.history.replaceState(
    {},
    document.title,
    `${callbackUrl.pathname}${callbackUrl.hash}`,
  );

  const transaction = parseOAuthTransaction(
    sessionStorage.getItem(OAUTH_STORAGE_KEY),
  );
  sessionStorage.removeItem(OAUTH_STORAGE_KEY);

  if (!transaction || !returnedState) {
    throw new Error("Google sign-in could not be completed. Please try again.");
  }
  if (
    Date.now() - transaction.createdAt > OAUTH_STATE_MAX_AGE_MS ||
    returnedState !== transaction.state
  ) {
    throw new Error("Google sign-in expired or failed its security check.");
  }
  if (providerError) {
    throw new Error("Google sign-in was cancelled or denied.");
  }
  if (!code) {
    throw new Error("Google did not return an authorization code.");
  }

  return exchangeGoogleCode(
    code,
    transaction.codeVerifier,
    transaction.nonce,
  );
}
