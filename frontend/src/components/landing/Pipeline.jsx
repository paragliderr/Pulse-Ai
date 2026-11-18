// This IS a genuine sequence (data flows through these stages in order),
// so numbering here is earning its place rather than decorating.
const STEPS = [
  { title: "Ingest", detail: "Prices, headlines, and fundamentals pulled from live market APIs." },
  { title: "Understand", detail: "Sentiment scoring, technical indicators, session and graph construction." },
  { title: "Learn", detail: "LightGCN for long-term collaborative signal, SR-GNN for in-session intent." },
  { title: "Explain", detail: "Every score gets a SHAP-based reason, not just a number." },
  { title: "Serve", detail: "A lightweight, compressed model behind a fast API, ready for a dashboard." },
];

export default function Pipeline() {
  return (
    <ol className="grid gap-8 sm:grid-cols-5">
      {STEPS.map((step, i) => (
        <li key={step.title} className="border-t border-white/15 pt-4">
          <span className="font-mono text-sm text-white/40">{String(i + 1).padStart(2, "0")}</span>
          <h4 className="mt-2 font-display text-xl text-white">{step.title}</h4>
          <p className="mt-1 text-sm leading-relaxed text-white/60">{step.detail}</p>
        </li>
      ))}
    </ol>
  );
}