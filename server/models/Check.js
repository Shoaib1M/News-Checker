/*
FILE PURPOSE:
Defines the Mongoose schema and model for a fact-check request (a "Check").
This stores the history of statements that users have submitted, along with the ML scores, evidence, and final verdicts.

FLOW:
1. Define a sub-schema for individual pieces of evidence (articles scraped from the web).
2. Define the main Check schema.
3. Export the Check model so we can save new checks and fetch user history.

USED BY:
- server/routes/check.js (to save a check to the database after the ML service responds)
- server/routes/history.js (to fetch a user's past checks)
*/

import mongoose from "mongoose";

/*
PURPOSE:
Sub-schema for an Evidence Item.
Since a single Check can have multiple pieces of evidence (articles), 
we define the structure of those items here.

WHY THIS EXISTS:
This keeps our main checkSchema clean and ensures every piece of evidence has a consistent structure.
*/
const evidenceItemSchema = new mongoose.Schema(
  {
    title: String,               // Headline of the article
    url: String,                 // Link to the article
    similarity: Number,          // How semantically similar the article is to the statement (0 to 1)
    stance: String,              // Does this article support, contradict, or is it neutral?
    source: String,              // The domain name (e.g., cnn.com)
    best_sentence: String,       // The specific sentence in the article that best matches the statement
    support_score: Number,       // NLI model's confidence that it supports
    contradiction_score: Number, // NLI model's confidence that it contradicts
    source_tier: String,         // primary, fact-check, reporting, or unclassified
    nli_available: Boolean,      // Whether the stance came from the NLI model
  },
  // _id: false tells Mongoose NOT to create a unique ID for every single piece of evidence.
  // We only need an ID for the parent Check document.
  { _id: false }
);

/*
PURPOSE:
The main schema for a fact-check record.
Stores what the user asked, what the ML model predicted, and the evidence found.
*/
const checkSchema = new mongoose.Schema(
  {
    // userId links this check to a specific User in our database.
    // This is how we implement the "History" feature.
    userId: {
      type: mongoose.Schema.Types.ObjectId, // The type is a MongoDB ID
      ref: "User",                          // It refers to the "User" model
      index: true,                          // Indexed for fast lookups when fetching a user's history
    },
    
    // The actual text the user submitted to be checked
    statement: {
      type: String,
      required: true,
    },
    
    // The raw probability score from our Binary Truth MLP model
    mlScore: {
      type: Number,
      required: true,
    },
    
    // The text label based on the mlScore (e.g., "Likely False")
    mlVerdict: String,
    
    // The overall score derived purely from the web evidence
    evidenceScore: Number,
    
    // A summary of the stance across all evidence articles
    evidenceStance: {
      support: Number,
      contradiction: Number,
      net: Number,
      verdict: String,
    },
    
    // The final blended score (ML model + Evidence) from 0 to 100
    combinedScore: {
      type: Number,
      required: true,
    },
    
    // The final text label shown to the user (e.g., "Very Likely False")
    combinedVerdict: String,

    // Evidence-first assessment metadata.  This lets saved results preserve
    // an explicit abstention instead of looking like a numerical verdict.
    assessmentStatus: String,
    claimAssessments: [{
      claim: String,
      status: String,
      verdict: String,
      support: Number,
      contradiction: Number,
      evidenceCount: Number,
    }],
    
    // An array of the top evidence articles found (uses the sub-schema defined above)
    topEvidence: [evidenceItemSchema],
    
    // How long it took the ML service to process this request (in seconds)
    processingTime: Number,
  },
  // Automatically adds createdAt and updatedAt dates
  { timestamps: true }
);

// Export the Check model. This connects to the "checks" collection in MongoDB.
export default mongoose.model("Check", checkSchema);
