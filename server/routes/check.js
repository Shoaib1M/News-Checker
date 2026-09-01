/*
FILE PURPOSE:
This route acts as a "Proxy" (a middleman) between the React Frontend and the Python ML Service.

FLOW:
1. Receives a statement from the frontend.
2. Forwards (proxies) that statement to the Python FastAPI service.
3. Receives the ML prediction and scraped evidence from Python.
4. If the user is logged in, saves the result to MongoDB.
5. Sends the final result back to the frontend.

USED BY:
- The main input form on the React frontend when a user clicks "Check Statement".
*/

import { Router } from "express";
import fetch from "node-fetch"; // Node doesn't have a built-in fetch (until recently), so we use this package
import Check from "../models/Check.js";
import { optionalAuth } from "../middleware/auth.js";

const router = Router();

// URL of the Python ML Service
// In development, this is usually http://localhost:8000
const FASTAPI_URL = process.env.FASTAPI_URL || "http://localhost:8000";

/*
PURPOSE:
Process a fact-check request.

INPUT:
req.body.statement (The string the user typed)

OUTPUT:
The combined results from the ML model and evidence scraper.

WHY THIS EXISTS:
Why not have the frontend talk directly to Python? 
Because we need to save the result in our MongoDB database to show the user's history. 
Our Node server handles the database, so the request must pass through here first.
*/
router.post("/", optionalAuth, async (req, res) => {
  try {
    const { statement } = req.body;

    // Step 1: Basic validation
    // Ensure the statement is a valid string and at least 5 characters long
    if (!statement || typeof statement !== "string" || statement.trim().length < 5) {
      return res.status(400).json({ error: "Statement must be at least 5 characters." });
    }

    // Step 2: Forward the request to the Python FastAPI service
    // We send a POST request to FastAPI, passing along the statement
    const mlResponse = await fetch(`${FASTAPI_URL}/api/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ statement: statement.trim() }),
    });

    // Step 3: Handle potential errors from the Python service
    if (!mlResponse.ok) {
      const errorText = await mlResponse.text();
      console.error("FastAPI error:", mlResponse.status, errorText);
      // 502 Bad Gateway means our server (Node) couldn't get a valid response from the upstream server (Python)
      return res.status(502).json({ error: "ML service error.", details: errorText });
    }

    // Step 4: Parse the JSON response from Python
    const result = await mlResponse.json();

    // Step 5: Save to MongoDB (ONLY if the user is logged in)
    let savedCheck = null;
    
    // req.user is populated by the `optionalAuth` middleware if a valid token was sent
    if (req.user) {
      try {
        // Create a new record in the 'checks' collection
        savedCheck = await Check.create({
          userId: req.user.userId,
          statement: result.statement,
          mlScore: result.ml_score,
          mlVerdict: result.ml_verdict,
          evidenceScore: result.evidence_score,
          evidenceStance: result.evidence_stance,
          combinedScore: result.combined_score,
          combinedVerdict: result.combined_verdict,
          assessmentStatus: result.assessment_status,
          claimAssessments: (result.claim_assessments || []).map((assessment) => ({
            claim: assessment.claim,
            status: assessment.status,
            verdict: assessment.verdict,
            support: assessment.support,
            contradiction: assessment.contradiction,
            evidenceCount: assessment.evidence_count,
          })),
          topEvidence: result.top_evidence,
          processingTime: result.processing_time_seconds,
        });
      } catch (dbError) {
        // If saving fails, we log it, but we DON'T crash the request.
        // We still want to show the user their result even if history failed to save.
        console.error("Failed to save check to DB:", dbError.message);
      }
    }

    // Step 6: Send the data back to the React frontend
    res.json({
      ...result, // Spread all properties from the Python result
      _id: savedCheck?._id || null, // Attach the MongoDB ID if we saved it (useful for UI updates)
    });
    
  } catch (error) {
    console.error("Check proxy error:", error.message);
    res.status(500).json({ error: "Failed to process fact-check request." });
  }
});

export default router;
