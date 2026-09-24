export default function FeedCard({ item }) {
  // item: { itemId, itemType, score, explanation }
  // explanation is expected to be a small object of {feature: contribution}
  // coming from the SHAP/LIME layer in the recommendation engine.
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          {item.itemType}
        </span>
        <span className="text-xs text-slate-400">score {item.score.toFixed(2)}</span>
      </div>
      <h3 className="mt-1 text-lg font-semibold">{item.itemId}</h3>

      {item.explanation && (
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(item.explanation).map(([feature, weight]) => (
            <span
              key={feature}
              className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-600"
              title={`contribution: ${weight}`}
            >
              {feature}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
