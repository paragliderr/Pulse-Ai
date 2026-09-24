import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import authRoutes from "./routes/auth.routes.js";
import watchlistRoutes from "./routes/watchlist.routes.js";
import feedRoutes from "./routes/feed.routes.js";

// Load the root .env regardless of which directory this is launched from
// (dotenv.config() with no path only checks process.cwd(), which breaks
// as soon as you `cd backend && npm run dev` instead of running from repo root).
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, "../../.env") });

const app = express();
const PORT = process.env.BACKEND_PORT || 4000;

app.use(cors());
app.use(express.json());

app.get("/health", (_req, res) => res.json({ status: "ok" }));

// Surface config problems immediately instead of failing mysteriously on
// the first request that needs them.
if (!process.env.DATABASE_URL) console.warn("[config] DATABASE_URL is not set — check .env");
if (!process.env.JWT_SECRET) console.warn("[config] JWT_SECRET is not set — check .env");
console.log(`[config] RECOMMENDATION_ENGINE_URL = ${process.env.RECOMMENDATION_ENGINE_URL || "(default) http://localhost:8000"}`);

// Don't let one bad request silently kill the whole process — log it and
// keep serving. Without this, an unhandled rejection deep in a dependency
// can drop the connection with no HTTP response at all, which the browser
// reports as a generic network error rather than a proper error status.
process.on("unhandledRejection", (reason) => console.error("[unhandledRejection]", reason));
process.on("uncaughtException", (err) => console.error("[uncaughtException]", err));

app.use("/api/auth", authRoutes);
app.use("/api/watchlist", watchlistRoutes);
app.use("/api/feed", feedRoutes);

app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(err.status || 500).json({ error: err.message || "Internal server error" });
});

app.listen(PORT, () => console.log(`Pulse AI backend listening on :${PORT}`));
