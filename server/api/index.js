import "dotenv/config";
import express from "express";
import cors from "cors";
import mongoose from "mongoose";

import authRoutes from "../routes/auth.js";
import checkRoutes from "../routes/check.js";
import historyRoutes from "../routes/history.js";

const app = express();
const PORT = process.env.PORT || 3001;

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------
app.use(cors({
  origin: process.env.CLIENT_URL || "http://localhost:5173",
  credentials: true,
}));
app.use(express.json({ limit: "1mb" }));

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------
app.use("/api/auth", authRoutes);
app.use("/api/check", checkRoutes);
app.use("/api/history", historyRoutes);

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", service: "express-proxy" });
});

// ---------------------------------------------------------------------------
// MongoDB connection + server start
// ---------------------------------------------------------------------------
const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017/newschecker";

mongoose
  .connect(MONGODB_URI)
  .then(() => {
    console.log("Connected to MongoDB.");
    app.listen(PORT, () => {
      console.log(`Express server running on http://localhost:${PORT}`);
    });
  })
  .catch((error) => {
    console.error("MongoDB connection failed:", error.message);
    // Start anyway so health checks work during development without MongoDB
    app.listen(PORT, () => {
      console.log(`Express server running on http://localhost:${PORT} (no database)`);
    });
  });

// Export for Vercel serverless
export default app;
