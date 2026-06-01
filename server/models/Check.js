import mongoose from "mongoose";

const evidenceItemSchema = new mongoose.Schema(
  {
    title: String,
    url: String,
    similarity: Number,
    stance: String,
    source: String,
    best_sentence: String,
    support_score: Number,
    contradiction_score: Number,
  },
  { _id: false }
);

const checkSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      index: true,
    },
    statement: {
      type: String,
      required: true,
    },
    mlScore: {
      type: Number,
      required: true,
    },
    mlVerdict: String,
    evidenceScore: Number,
    evidenceStance: {
      support: Number,
      contradiction: Number,
      net: Number,
      verdict: String,
    },
    combinedScore: {
      type: Number,
      required: true,
    },
    combinedVerdict: String,
    topEvidence: [evidenceItemSchema],
    processingTime: Number,
  },
  { timestamps: true }
);

export default mongoose.model("Check", checkSchema);
