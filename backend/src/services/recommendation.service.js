import axios from "axios";

const RECO_URL = process.env.RECOMMENDATION_ENGINE_URL || "http://localhost:8000";

/**
 * Calls the recommendation engine's /recommend endpoint.
 * The engine returns a ranked list of {itemId, itemType, score, explanation}.
 * Backend is responsible for auth + caching (see FeedCache in schema.prisma);
 * the recommendation engine itself is stateless about who's asking beyond
 * the userId/watchlist it's given.
 */
export async function getRecommendations({ userId, watchlist, topK = 20 }) {
  const { data } = await axios.post(`${RECO_URL}/recommend`, {
    user_id: userId,
    watchlist,
    top_k: topK,
  });
  return data.recommendations;
}
