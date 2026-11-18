// The 40 tickers the engine serves, by sector.
export const SECTORS = {
  Technology: ["AAPL", "MSFT", "NVDA", "AMD", "ROKU"],
  Healthcare: ["LLY", "JNJ", "PFE", "GILD", "MRNA"],
  Financials: ["JPM", "BAC", "GS", "SCHW", "ALLY"],
  Energy: ["XOM", "CVX", "COP", "OXY", "APA"],
  Consumer: ["AMZN", "WMT", "NKE", "SBUX", "ETSY"],
  Industrials: ["GE", "CAT", "UPS", "DAL", "AAL"],
  Utilities: ["NEE", "DUK", "SO", "AEP", "PNW"],
  Materials: ["LIN", "SHW", "FCX", "NEM", "CLF"],
};
export const ALL_TICKERS = Object.values(SECTORS).flat();
