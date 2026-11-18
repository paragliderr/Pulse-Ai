"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import api from "../../lib/api";
import FeedCard from "../../components/FeedCard";
import Nav from "../../components/Nav";

const LABEL = {
  session: "Based on what you just read",
  watchlist: "Based on your watchlist",
  generic: "Market-wide feed",
};

export default function DashboardPage() {
  const [items, setItems] = useState([]);
  const [meta, setMeta] = useState({});
  const [status, setStatus] = useState(null);
  const [sector, setSector] = useState("All");
  const [error, setError] = useState(null);
  const [loggedIn, setLoggedIn] = useState(false);
  const demoRecent = useRef([]); // clicks made while logged out

  async function load() {
    const token = localStorage.getItem("pulse_token");
    setLoggedIn(!!token);
    const url = token ? "/api/feed" : `/api/feed/demo?recent=${demoRecent.current.join(",")}`;
    try {
      const res = await api.get(url);
      setItems(res.data.items);
      setMeta(res.data);
      setError(null);
    } catch (err) {
      if (err.response?.status === 401) localStorage.removeItem("pulse_token");
      setError(err.response?.data?.error || err.message);
    }
  }

  useEffect(() => {
    load();
    api.get("/api/status").then((r) => setStatus(r.data)).catch(() => setStatus({ engine: "down" }));
  }, []);

  function log(item, action) {
    if (loggedIn) return api.post("/api/interactions", { itemId: item.itemId, itemType: item.itemType, action });
  }

  async function onRead(item) {
    if (item.url) window.open(item.url, "_blank");
    if (loggedIn) await log(item, "CLICK");
    else demoRecent.current = [...demoRecent.current, item.itemId].slice(-10);
    load(); // re-rank the feed
  }

  const sectors = ["All", ...new Set(items.map((i) => i.sector))];
  const shown = sector === "All" ? items : items.filter((i) => i.sector === sector);

  return (
    <div className="min-h-screen bg-ink text-white">
      <Nav />
      <main className="mx-auto max-w-3xl px-4 pb-16">
        <h1 className="font-display text-4xl">Your Pulse</h1>
        <p className="mt-2 text-white/60">
          {LABEL[meta.personalisation] || "Loading..."}
          {meta.catalogueMode === "sample" && " · placeholder headlines"}
        </p>

        {!loggedIn && (
          <p className="mt-4 rounded-lg bg-brass/15 px-4 py-2 text-sm text-brass">
            Demo feed. <Link href="/signup" className="underline">Sign up</Link> to use a watchlist and save your reading.
          </p>
        )}
        {loggedIn && meta.personalisation === "generic" && (
          <p className="mt-4 rounded-lg bg-white/10 px-4 py-2 text-sm text-white/70">
            Add tickers in your <Link href="/watchlist" className="underline">watchlist</Link> to personalise this feed.
          </p>
        )}
        {error && <p className="mt-4 rounded-lg bg-signal-red/20 px-4 py-2 text-sm text-signal-red">{error}</p>}

        <div className="mt-6 flex flex-wrap gap-2 text-xs">
          {sectors.map((s) => (
            <button key={s} onClick={() => setSector(s)}
              className={`rounded-full border px-3 py-1 ${sector === s ? "border-brass text-brass" : "border-white/20 text-white/60"}`}>
              {s}
            </button>
          ))}
          <button onClick={load} className="ml-auto rounded-full border border-white/20 px-3 py-1 text-white/60">Refresh</button>
        </div>

        <div className="mt-4 flex flex-col gap-4">
          {shown.map((item) => <FeedCard key={item.itemId} item={item} onRead={onRead} onView={(i) => log(i, "VIEW")} />)}
          {!error && shown.length === 0 && <p className="text-white/50">No stories to show.</p>}
        </div>

        {status && (
          <p className="mt-10 text-center font-mono text-xs text-white/30">
            backend ok · engine {status.engine}{status.model && ` · ${status.model.type}`}
          </p>
        )}
      </main>
    </div>
  );
}
