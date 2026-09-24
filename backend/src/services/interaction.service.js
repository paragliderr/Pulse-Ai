import prisma from "../config/prisma.js";

/**
 * Logs a user interaction event (view/click/dwell/watchlist change).
 * This is the raw signal the recommendation engine consumes for graph
 * and session construction — see recommendation-engine/app/data/preprocessing.
 *
 * For simplicity this creates a new session per call if none is open;
 * a production version would track an active session per user with a
 * time-gap threshold (matches the "session segmentation" preprocessing
 * step described in the Milestone 2 report).
 */
export async function logInteraction({ userId, itemId, itemType, action }) {
  let session = await prisma.session.findFirst({
    where: { userId, endedAt: null },
    orderBy: { startedAt: "desc" },
  });

  if (!session) {
    session = await prisma.session.create({ data: { userId } });
  }

  return prisma.interaction.create({
    data: { userId, sessionId: session.id, itemId, itemType, action },
  });
}
