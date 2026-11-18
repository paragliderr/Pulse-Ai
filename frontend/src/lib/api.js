import axios from "axios";

// No host here on purpose — requests go to this app's own /api/*, which
// next.config.js rewrites (proxies) to the real backend server-side. This
// avoids the browser ever needing to resolve the backend's host directly,
// which is what broke in cloud/sandboxed dev environments where "localhost"
// in the browser doesn't map to the backend's actual address.
const api = axios.create({
  baseURL: "",
});

// Attach the JWT from localStorage to every request, once auth is wired up client-side.
api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("pulse_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
