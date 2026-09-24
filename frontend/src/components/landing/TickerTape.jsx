const TICKERS = [
  { symbol: "AAPL", delta: "+0.84%", up: true },
  { symbol: "MSFT", delta: "-0.21%", up: false },
  { symbol: "NVDA", delta: "+2.13%", up: true },
  { symbol: "TSLA", delta: "-1.05%", up: false },
  { symbol: "JPM", delta: "+0.32%", up: true },
  { symbol: "META", delta: "+0.58%", up: true },
  { symbol: "V", delta: "-0.09%", up: false },
];

export default function TickerTape() {
  const row = [...TICKERS, ...TICKERS]; // duplicated for a seamless loop
  return (
    <div className="overflow-hidden border-y border-white/10 py-3">
      <div className="flex w-max animate-ticker gap-10">
        {row.map((t, i) => (
          <span key={i} className="flex items-center gap-2 font-mono text-sm text-white/70">
            {t.symbol}
            <span className={t.up ? "text-signal-green" : "text-signal-red"}>{t.delta}</span>
          </span>
        ))}
      </div>
    </div>
  );
}