import { createContext, useContext, useEffect, useState } from "react";
import { me } from "./api.js";

const TOKEN_KEY = "ada_token";

// --- raw token accessors (kept React-free so api.js can import without a cycle) ---
export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore storage errors (private mode, etc.) */
  }
}

export const AuthContext = createContext({
  token: null,
  user: null,
  login: () => {},
  logout: () => {}
});

export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(() => getToken());
  const [user, setUser] = useState(null);

  // On mount (or when a persisted token exists) hydrate the current user.
  useEffect(() => {
    let cancelled = false;
    if (token && !user) {
      me()
        .then((u) => {
          if (!cancelled) setUser(u);
        })
        .catch(() => {
          // Token invalid/expired — clear it.
          if (!cancelled) {
            setToken(null);
            setTokenState(null);
            setUser(null);
          }
        });
    }
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  function login(newToken, newUser) {
    setToken(newToken);
    setTokenState(newToken);
    if (newUser) setUser(newUser);
  }

  function logout() {
    setToken(null);
    setTokenState(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ token, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
