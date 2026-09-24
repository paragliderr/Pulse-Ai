import prisma from "../config/prisma.js";
import { logInteraction } from "../services/interaction.service.js";

export async function getWatchlist(req, res, next) {
  try {
    const items = await prisma.watchlistItem.findMany({ where: { userId: req.userId } });
    res.json(items);
  } catch (err) {
    next(err);
  }
}

export async function addToWatchlist(req, res, next) {
  try {
    const { ticker } = req.body;
    if (!ticker) return res.status(400).json({ error: "ticker is required" });

    const item = await prisma.watchlistItem.create({
      data: { userId: req.userId, ticker: ticker.toUpperCase() },
    });

    await logInteraction({ userId: req.userId, itemId: ticker, itemType: "ticker", action: "WATCHLIST_ADD" });
    res.status(201).json(item);
  } catch (err) {
    next(err);
  }
}

export async function removeFromWatchlist(req, res, next) {
  try {
    const { ticker } = req.params;
    await prisma.watchlistItem.delete({
      where: { userId_ticker: { userId: req.userId, ticker: ticker.toUpperCase() } },
    });
    await logInteraction({ userId: req.userId, itemId: ticker, itemType: "ticker", action: "WATCHLIST_REMOVE" });
    res.status(204).send();
  } catch (err) {
    next(err);
  }
}
