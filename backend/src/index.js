import crypto from "crypto";
import express from "express";
import cors from "cors";
import fs from "fs";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";

const PORT = process.env.PORT || 4000;
const ENGINE = process.env.ENGINE_URL || "http://localhost:8000";
const SECRET = process.env.JWT_SECRET || "dev-secret-change-me";
const DB_FILE = new URL("../data.json", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

// ---- tiny JSON-file "database": { users: [], watchlist: [], interactions: [] } ----
const db = fs.existsSync(DB_FILE)
  ? JSON.parse(fs.readFileSync(DB_FILE, "utf8"))
  : { users: [], watchlist: [], interactions: [] };
const save = () => fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2));

const app = express();
app.use(cors());
app.use(express.json());

// ---- helpers ----
const wrap = (fn) => (req, res) => Promise.resolve(fn(req, res)).catch((e) => res.status(500).json({ error: e.message }));
const list = (s) => (s ? String(s).split(",").map((x) => x.trim()).filter(Boolean) : []);

function auth(req, res, next) {
  try {
    req.userId = jwt.verify((req.headers.authorization || "").replace("Bearer ", ""), SECRET).userId;
    next();
  } catch {
    res.status(401).json({ error: "Please log in" });
  }
}

// ask the engine and rename fields for the frontend
async function getFeed(watchlist, recent) {
  let r;
  try {
    r = await fetch(`${ENGINE}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ watchlist, recent_items: recent, top_k: 10 }),
    });
  } catch {
    throw new Error("Recommendation engine is not running on " + ENGINE);
  }
  if (!r.ok) throw new Error("Engine error " + r.status);
  const d = await r.json();
  return {
    personalisation: d.personalisation,
    catalogueMode: d.catalogue_mode,
    items: d.recommendations.map((i) => ({
      itemId: i.item_id, itemType: i.item_type, ticker: i.ticker, sector: i.sector,
      headline: i.headline, source: i.source, url: i.url, score: i.score,
      reasons: i.explanation.reasons, features: i.explanation.features,
    })),
  };
}

// ---- routes ----
app.get("/health", (_q, res) => res.json({ status: "ok" }));

// status of every service, for the frontend status badge
app.get("/api/status", wrap(async (_q, res) => {
  try {
    const [h, m] = await Promise.all([fetch(`${ENGINE}/health`), fetch(`${ENGINE}/model/info`)]);
    res.json({ backend: "ok", engine: (await h.json()).status, model: await m.json() });
  } catch {
    res.json({ backend: "ok", engine: "down", model: null });
  }
}));

function authRoute(isSignup) {
  return wrap((req, res) => {
    const { email, password } = req.body || {};
    if (!email || !email.includes("@")) return res.status(400).json({ error: "Enter a valid email" });
    if (!password || password.length < 8) return res.status(400).json({ error: "Password must be at least 8 characters" });
    let user = db.users.find((u) => u.email === email.toLowerCase());
    if (isSignup) {
      if (user) return res.status(409).json({ error: "Email already registered" });
      user = { id: crypto.randomUUID(), email: email.toLowerCase(), hash: bcrypt.hashSync(password, 10) };
      db.users.push(user);
      save();
    } else if (!user || !bcrypt.compareSync(password, user.hash)) {
      return res.status(401).json({ error: "Invalid email or password" });
    }
    const token = jwt.sign({ userId: user.id }, SECRET, { expiresIn: "7d" });
    res.status(isSignup ? 201 : 200).json({ token, user: { id: user.id, email: user.email } });
  });
}
app.post("/api/auth/signup", authRoute(true));
app.post("/api/auth/login", authRoute(false));

app.get("/api/watchlist", auth, (req, res) =>
  res.json(db.watchlist.filter((w) => w.userId === req.userId).map((w) => ({ ticker: w.ticker }))));

app.post("/api/watchlist", auth, (req, res) => {
  const ticker = String(req.body?.ticker || "").toUpperCase().trim();
  if (!ticker) return res.status(400).json({ error: "ticker is required" });
  if (!db.watchlist.some((w) => w.userId === req.userId && w.ticker === ticker)) {
    db.watchlist.push({ userId: req.userId, ticker });
    save();
  }
  res.status(201).json({ ticker });
});

app.delete("/api/watchlist/:ticker", auth, (req, res) => {
  db.watchlist = db.watchlist.filter((w) => !(w.userId === req.userId && w.ticker === req.params.ticker.toUpperCase()));
  save();
  res.status(204).send();
});

// demo feed: no login, e.g. /api/feed/demo?recent=NEE:0,DUK:1&watchlist=XOM,COP
app.get("/api/feed/demo", wrap(async (req, res) => res.json(await getFeed(list(req.query.watchlist), list(req.query.recent)))));

// personalised feed: watchlist + last 10 clicks (oldest first), no cache
app.get("/api/feed", auth, wrap(async (req, res) => {
  const watchlist = db.watchlist.filter((w) => w.userId === req.userId).map((w) => w.ticker);
  const recent = db.interactions
    .filter((i) => i.userId === req.userId && i.action === "CLICK")
    .slice(-10).map((i) => i.itemId);
  res.json(await getFeed(watchlist, recent));
}));

app.post("/api/interactions", auth, (req, res) => {
  const { itemId, itemType, action } = req.body || {};
  if (!itemId || !["VIEW", "CLICK"].includes(action)) return res.status(400).json({ error: "itemId and action (VIEW|CLICK) required" });
  db.interactions.push({ userId: req.userId, itemId, itemType: itemType || "news", action, at: new Date().toISOString() });
  save();
  res.status(201).json({ ok: true });
});

app.listen(PORT, () => console.log(`Backend on :${PORT}  (engine: ${ENGINE})`));
