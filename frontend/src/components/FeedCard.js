"use client";
import { useState } from "react";

export default function FeedCard({ item, onRead, onView }) {
  const [open, setOpen] = useState(false);

  function toggle() {
    if (!open) onView(item); // log a VIEW the first time they look at the explanation
    setOpen(!open);
  }

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
      <div className="flex items-center justify-between text-xs text-white/50">
        <span className="font-mono uppercase">
          {item.ticker} · {item.sector}
          <span className={`ml-2 rounded px-1.5 py-0.5 ${item.source === "live" ? "bg-signal-green/20 text-signal-green" : "bg-brass/20 text-brass"}`}>
            {item.source === "live" ? "live" : "sample"}
          </span>
        </span>
        <span className="font-mono">score {item.score.toFixed(2)}</span>
      </div>

      <h3 className="mt-2 font-display text-xl text-white">{item.headline}</h3>

      <div className="mt-3 flex flex-wrap gap-2">
        {item.reasons.map((r) => (
          <span key={r} className="rounded-full bg-white/10 px-3 py-1 text-xs text-white/70">{r}</span>
        ))}
      </div>

      {open && (
        <div className="mt-4 space-y-2 border-t border-white/10 pt-4">
          {Object.entries(item.features).map(([name, v]) => (
            <div key={name} className="flex items-center gap-3 text-xs text-white/60">
              <span className="w-20 font-mono">{name.replace("_", " ")}</span>
              <div className="h-1.5 flex-1 rounded bg-white/10">
                <div className="h-1.5 rounded bg-brass" style={{ width: `${Math.min(Math.abs(v), 1) * 100}%` }} />
              </div>
              <span className="w-10 text-right font-mono">{v.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 flex gap-3 text-xs">
        <button onClick={() => onRead(item)} className="rounded-full bg-brass px-4 py-1.5 font-medium text-ink">Read</button>
        <button onClick={toggle} className="rounded-full border border-white/20 px-4 py-1.5 text-white/70 hover:border-white/50">
          {open ? "Hide why" : "Why this?"}
        </button>
      </div>
    </div>
  );
}
