import { PrismaClient } from "@prisma/client";

// Reuse a single client instance across the app (recommended by Prisma docs
// to avoid exhausting DB connections during dev hot-reload).
const prisma = globalThis.__prisma__ ?? new PrismaClient();
if (process.env.NODE_ENV !== "production") globalThis.__prisma__ = prisma;

export default prisma;
