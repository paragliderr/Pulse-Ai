"use client";

import { useEffect, useState } from "react";
import api from "../../lib/api";
import FeedCard from "../../components/FeedCard";

export default function DashboardPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isDemo, setIsDemo] = useState(false);

  useEffect(() => {
    const token = typeof window !== "undefined" ? localStorage.getItem("pulse_token") : null;
    const endpoint = token ? "/api/feed" : "/api/feed/demo";
    setIsDemo(!token);

    api
      .get(endpoint)
      .then((res) => setItems(res.data.items))
      .catch((err) => setError(err.response?.data?.error || err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <main className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="text-2xl font-bold">Your Pulse</h1>
      <p className="mt-1 text-sm text-slate-500">
        Personalized news and market signals, with why each one is here.
      </p>

      {isDemo && !loading && !error && (
        <p className="mt-3 inline-block rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-700">
          Showing demo data — log in for your personalized feed
        </p>
      )}

      {loading && <p className="mt-6 text-slate-500">Loading feed…</p>}
      {error && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-600">
          <p className="font-medium">Couldn't load feed: {error}</p>
        </div>
      )}

      <div className="mt-6 flex flex-col gap-3">
        {items.map((item) => (
          <FeedCard key={`${item.itemType}-${item.itemId}`} item={item} />
        ))}
      </div>
    </main>
  );
}