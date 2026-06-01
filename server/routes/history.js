import { Router } from "express";
import Check from "../models/Check.js";
import { requireAuth } from "../middleware/auth.js";

const router = Router();

/**
 * GET /api/history
 * Returns the authenticated user's past fact-checks, newest first.
 * Query params: ?limit=50&skip=0
 */
router.get("/", requireAuth, async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 50, 100);
    const skip = parseInt(req.query.skip) || 0;

    const checks = await Check.find({ userId: req.user.userId })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .select("statement combinedScore combinedVerdict createdAt")
      .lean();

    const total = await Check.countDocuments({ userId: req.user.userId });

    res.json({ checks, total, limit, skip });
  } catch (error) {
    console.error("History fetch error:", error.message);
    res.status(500).json({ error: "Failed to fetch history." });
  }
});

/**
 * GET /api/history/:id
 * Returns the full detail for a single past fact-check.
 */
router.get("/:id", requireAuth, async (req, res) => {
  try {
    const check = await Check.findOne({
      _id: req.params.id,
      userId: req.user.userId,
    }).lean();

    if (!check) {
      return res.status(404).json({ error: "Check not found." });
    }

    res.json(check);
  } catch (error) {
    console.error("History detail error:", error.message);
    res.status(500).json({ error: "Failed to fetch check." });
  }
});

/**
 * DELETE /api/history/:id
 * Deletes a single fact-check from the user's history.
 */
router.delete("/:id", requireAuth, async (req, res) => {
  try {
    const result = await Check.deleteOne({
      _id: req.params.id,
      userId: req.user.userId,
    });

    if (result.deletedCount === 0) {
      return res.status(404).json({ error: "Check not found." });
    }

    res.json({ message: "Deleted successfully." });
  } catch (error) {
    console.error("History delete error:", error.message);
    res.status(500).json({ error: "Failed to delete check." });
  }
});

export default router;
