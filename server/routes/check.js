import { Router } from "express";
import fetch from "node-fetch";
import Check from "../models/Check.js";
import { optionalAuth } from "../middleware/auth.js";

const router = Router();
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";

/**
 * POST /api/check
 * Body: { statement: "..." }
 *
 * Proxies the request to the FastAPI ML service, then saves the result
 * to MongoDB if the user is authenticated.
 */
router.post("/", optionalAuth, async (req, res) => {
  try {
    const { statement } = req.body;

    if (!statement || typeof statement !== "string" || statement.trim().length < 5) {
      return res.status(400).json({ error: "Statement must be at least 5 characters." });
    }

    // Forward to FastAPI
    const mlResponse = await fetch(`${FASTAPI_URL}/api/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ statement: statement.trim() }),
    });

    if (!mlResponse.ok) {
      const errorText = await mlResponse.text();
      console.error("FastAPI error:", mlResponse.status, errorText);
      return res.status(502).json({ error: "ML service error.", details: errorText });
    }

    const result = await mlResponse.json();

    // Save to MongoDB if user is logged in
    let savedCheck = null;
    if (req.user) {
      try {
        savedCheck = await Check.create({
          userId: req.user.userId,
          statement: result.statement,
          mlScore: result.ml_score,
          mlVerdict: result.ml_verdict,
          evidenceScore: result.evidence_score,
          evidenceStance: result.evidence_stance,
          combinedScore: result.combined_score,
          combinedVerdict: result.combined_verdict,
          topEvidence: result.top_evidence,
          processingTime: result.processing_time_seconds,
        });
      } catch (dbError) {
        console.error("Failed to save check to DB:", dbError.message);
      }
    }

    res.json({
      ...result,
      _id: savedCheck?._id || null,
    });
  } catch (error) {
    console.error("Check proxy error:", error.message);
    res.status(500).json({ error: "Failed to process fact-check request." });
  }
});

export default router;
