/*
FILE PURPOSE:
This file handles fetching and deleting the fact-check history for a logged-in user.

FLOW:
1. Defines routes to GET multiple checks (pagination supported).
2. Defines a route to GET a single check by its ID.
3. Defines a route to DELETE a single check.

USED BY:
- The React frontend's History Panel and individual result views.
*/

import { Router } from "express";
import Check from "../models/Check.js";
import { requireAuth } from "../middleware/auth.js";

const router = Router();

/*
PURPOSE:
Fetch a list of the user's past fact-checks.

INPUT:
URL Query parameters: ?limit=50&skip=0 (used for pagination)
Requires authentication.

OUTPUT:
An array of simplified check objects, total count, limit, and skip.

WHY THIS EXISTS:
To populate the user's dashboard/history panel. We only return a summary (statement, score, date)
to keep the payload small and fast.
*/
router.get("/", requireAuth, async (req, res) => {
  try {
    // Step 1: Parse pagination parameters from the URL
    // Limit how many items we return (max 100 to prevent database strain)
    const limit = Math.min(parseInt(req.query.limit) || 50, 100);
    // Skip determines how many items to skip (for page 2, 3, etc.)
    const skip = parseInt(req.query.skip) || 0;

    // Step 2: Query the database
    const checks = await Check.find({ userId: req.user.userId }) // Only get checks belonging to THIS user
      .sort({ createdAt: -1 })                                   // Sort by newest first (-1 means descending)
      .skip(skip)
      .limit(limit)
      .select("statement combinedScore combinedVerdict assessmentStatus createdAt") // Only fetch these specific fields
      .lean(); // .lean() converts Mongoose documents to plain JavaScript objects for faster performance

    // Step 3: Count total documents (needed by frontend to calculate total number of pages)
    const total = await Check.countDocuments({ userId: req.user.userId });

    // Step 4: Return the data
    res.json({ checks, total, limit, skip });
  } catch (error) {
    console.error("History fetch error:", error.message);
    res.status(500).json({ error: "Failed to fetch history." });
  }
});

/*
PURPOSE:
Fetch the full details of a specific past fact-check.

INPUT:
req.params.id (The MongoDB ID in the URL, e.g., /api/history/60d5ec...)

OUTPUT:
The complete check object including all evidence arrays.

WHY THIS EXISTS:
When a user clicks on an item in their history list, they want to see the full breakdown and evidence again.
*/
router.get("/:id", requireAuth, async (req, res) => {
  try {
    // Step 1: Find the specific check.
    // CRITICAL: We include userId: req.user.userId in the query!
    // This ensures a user cannot fetch someone else's history by guessing their ID.
    const check = await Check.findOne({
      _id: req.params.id,
      userId: req.user.userId,
    }).lean();

    // Step 2: Handle not found
    if (!check) {
      return res.status(404).json({ error: "Check not found." });
    }

    // Step 3: Return the full document
    res.json(check);
  } catch (error) {
    console.error("History detail error:", error.message);
    res.status(500).json({ error: "Failed to fetch check." });
  }
});

/*
PURPOSE:
Delete a specific fact-check from the user's history.

INPUT:
req.params.id

OUTPUT:
A success message.

WHY THIS EXISTS:
Allows users to manage their data and remove items they no longer want stored.
*/
router.delete("/:id", requireAuth, async (req, res) => {
  try {
    // Step 1: Attempt to delete the document.
    // Again, ensure it belongs to the requesting user.
    const result = await Check.deleteOne({
      _id: req.params.id,
      userId: req.user.userId,
    });

    // Step 2: Check if anything was actually deleted
    if (result.deletedCount === 0) {
      return res.status(404).json({ error: "Check not found." });
    }

    // Step 3: Success response
    res.json({ message: "Deleted successfully." });
  } catch (error) {
    console.error("History delete error:", error.message);
    res.status(500).json({ error: "Failed to delete check." });
  }
});

export default router;
