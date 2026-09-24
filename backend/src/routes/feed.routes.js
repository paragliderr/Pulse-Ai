import { Router } from "express";
import { requireAuth } from "../middleware/auth.middleware.js";
import { getFeed, getDemoFeed } from "../controllers/feed.controller.js";

const router = Router();

// GET /api/feed/demo -> no auth required, always returns 200 with data
// (falls back to inline mock data if the recommendation engine isn't
// running). This is what the frontend hits before a user logs in, and
// what frontend/backend teammates use to build against without needing
// the full stack up.
router.get("/demo", getDemoFeed);

// GET /api/feed -> personalized feed for the logged-in user (or a generic
// feed if they have no watchlist yet — see recommendation.service.js)
router.get("/", requireAuth, getFeed);

export default router;
