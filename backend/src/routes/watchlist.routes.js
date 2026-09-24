import { Router } from "express";
import { requireAuth } from "../middleware/auth.middleware.js";
import { getWatchlist, addToWatchlist, removeFromWatchlist } from "../controllers/watchlist.controller.js";

const router = Router();

router.get("/", requireAuth, getWatchlist);
router.post("/", requireAuth, addToWatchlist);
router.delete("/:ticker", requireAuth, removeFromWatchlist);

export default router;
