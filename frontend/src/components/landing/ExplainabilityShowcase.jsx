const SAMPLE = {
  itemId: "NVDA",
  itemType: "ticker",
  score: 0.91,
  explanation: { sentiment_spike: 0.42, sector_momentum: 0.31, similarity_to_watchlist: 0.19 },
};

export default function ExplainabilityShowcase() {
  return (
    <div className="max-w-md rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm">
      <div className="flex items-center justify-between text-sm text-white/40">
        <span>{SAMPLE.itemType}</span>
        <span className="font-mono text-white/60">score {SAMPLE.score.toFixed(2)}</span>
      </div>
      <h3 className="mt-2 font-display text-3xl text-white">{SAMPLE.itemId}</h3>
      <p className="mt-1 text-sm text-white/50">Surfaced because of:</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {Object.entries(SAMPLE.explanation).map(([feature, weight]) => (
          <span key={feature} className="rounded-full bg-white/10 px-3 py-1 font-mono text-xs text-white/70">
            {feature.replaceAll("_", " ")} · {weight.toFixed(2)}
          </span>
        ))}
      </div>
    </div>
  );
}