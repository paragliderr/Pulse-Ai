"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import api from "../../lib/api";
import Nav from "../../components/Nav";
import { SECTORS, ALL_TICKERS } from "../../lib/tickers";

export default function WatchlistPage() {
  const router = useRouter();
  const [tickers, setTickers] = useState([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState(null);

  async function load() {
    try {
      const res = await api.get("/api/watchlist");
      setTickers(res.data.map((w) => w.ticker));
    } catch {
      router.push("/login");
    }
  }
  useEffect(() => { load(); }, []);

  async function add(ticker) {
    ticker = ticker.toUpperCase().trim();
    if (!ALL_TICKERS.includes(ticker)) return setError(`${ticker} is not one of the 40 supported tickers`);
    try {
      await api.post("/api/watchlist", { ticker });
      setError(null);
      setInput("");
      load();
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    }
  }

  async function remove(t) {
    await api.delete(`/api/watchlist/${t}`);
    load();
  }

  return (
    <div className="min-h-screen bg-ink text-white">
      <Nav />
      <main className="mx-auto max-w-3xl px-4 pb-16">
        <h1 className="font-display text-4xl">Watchlist</h1>
        <p className="mt-2 text-white/60">Your feed starts from these tickers until you read something.</p>

        <form onSubmit={(e) => { e.preventDefault(); add(input); }} className="mt-6 flex gap-2">
          <input list="tickers" className="flex-1 rounded-lg border border-white/15 bg-white/5 p-3 uppercase outline-none focus:border-brass"
                 placeholder="Type a ticker, e.g. XOM" value={input} onChange={(e) => setInput(e.target.value)} />
          <datalist id="tickers">{ALL_TICKERS.map((t) => <option key={t} value={t} />)}</datalist>
          <button className="rounded-full bg-brass px-6 font-medium text-ink">Add</button>
        </form>
        {error && <p className="mt-2 text-sm text-signal-red">{error}</p>}

        <h2 className="mt-8 text-sm uppercase tracking-wide text-white/40">Following ({tickers.length})</h2>
        <div className="mt-3 flex flex-wrap gap-2">
          {tickers.map((t) => (
            <button key={t} onClick={() => remove(t)} title="Remove"
              className="rounded-full bg-brass/20 px-4 py-1.5 font-mono text-sm text-brass">{t} ✕</button>
          ))}
          {tickers.length === 0 && <p className="text-sm text-white/50">Nothing yet. Pick some below.</p>}
        </div>

        <h2 className="mt-10 text-sm uppercase tracking-wide text-white/40">All tickers</h2>
        {Object.entries(SECTORS).map(([sector, list]) => (
          <div key={sector} className="mt-4">
            <p className="text-sm text-white/60">{sector}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {list.map((t) => (
                <button key={t} disabled={tickers.includes(t)} onClick={() => add(t)}
                  className="rounded-full border border-white/20 px-3 py-1 font-mono text-xs text-white/70 enabled:hover:border-brass disabled:opacity-30">
                  {t}
                </button>
              ))}
            </div>
          </div>
        ))}

        <Link href="/dashboard" className="mt-10 inline-block rounded-full border border-white/20 px-6 py-2 text-sm">See my feed</Link>
      </main>
    </div>
  );
}
