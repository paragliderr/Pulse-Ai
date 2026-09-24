import prisma from "../config/prisma.js";
import { getRecommendations } from "../services/recommendation.service.js";

const CACHE_TTL_MINUTES = 15;

// Used when the recommendation engine isn't reachable, so the demo endpoint
// never hard-fails — frontend/backend work can proceed without the Python
// service running.
const FALLBACK_DEMO_ITEMS = [
  { item_id: "AAPL", item_type: "ticker", score: 0.91, explanation: { sector_momentum: 0.4, sentiment_spike: 0.3 } },
  { item_id: "MSFT", item_type: "ticker", score: 0.87, explanation: { price_momentum: 0.35, persona_match: 0.25 } },
  { item_id: "NVDA", item_type: "ticker", score: 0.82, explanation: { sentiment_spike: 0.5 } },
  { item_id: "Fed signals rate pause", item_type: "news", score: 0.78, explanation: { sector_momentum: 0.3 } },
];

/**
 * Public demo feed — no auth, no watchlist required. Tries the real
 * recommendation engine first; if it's not running (or errors), falls back
 * to static demo data so this endpoint always returns 200. This is the
 * "basic endpoint" for teammates to build the landing page and feed UI
 * against before auth and the ML service are fully wired up.
 */
export async function getDemoFeed(_req, res) {
  try {
    const recommendations = await getRecommendations({
      userId: "demo-user",
      watchlist: ["AAPL", "MSFT", "NVDA"],
      topK: 10,
    });
    return res.json({ source: "engine", items: recommendations });
  } catch (err) {
    console.warn("[demo feed] recommendation engine unreachable, using fallback data:", err.message);
    return res.json({ source: "fallback", items: FALLBACK_DEMO_ITEMS });
  }
}

export async function getFeed(req, res, next) {
  try {
    const watchlist = await prisma.watchlistItem.findMany({ where: { userId: req.userId } });
    const tickers = watchlist.map((w) => w.ticker);

    const cached = await prisma.feedCache.findMany({
      where: {
        userId: req.userId,
        generatedAt: { gte: new Date(Date.now() - CACHE_TTL_MINUTES * 60 * 1000) },
      },
      orderBy: { score: "desc" },
    });

    if (cached.length > 0) {
      return res.json({ source: "cache", items: cached });
    }

    const recommendations = await getRecommendations({ userId: req.userId, watchlist: tickers });

    await prisma.feedCache.createMany({
      data: recommendations.map((r) => ({
        userId: req.userId,
        itemId: r.item_id,
        itemType: r.item_type,
        score: r.score,
        explanation: r.explanation,
      })),
    });

    res.json({ source: "live", items: recommendations });
  } catch (err) {
    next(err);
  }
}
