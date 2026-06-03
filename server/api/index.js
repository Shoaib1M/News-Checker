/*
FILE PURPOSE:
This is the main entry point for the Node.js / Express backend server.
It sets up the web server, connects to the database (MongoDB), and routes incoming requests to the appropriate handlers.

FLOW:
1. Load environment variables.
2. Initialize Express application.
3. Apply middleware (CORS for cross-origin requests, JSON parser for request bodies).
4. Register API routes (Auth, Check, History).
5. Connect to MongoDB database.
6. Start listening for incoming network requests on a specified PORT.

USED BY:
- The React Frontend (Client) sends all its API requests to this server.
- Used as the serverless entry point for deployments (like Vercel).
*/

// Load environment variables from a .env file into process.env
import "dotenv/config";

// Express is the web framework we use to build the API
import express from "express";

// CORS (Cross-Origin Resource Sharing) allows our frontend (running on a different port/domain) to talk to this backend
import cors from "cors";

// Mongoose is an ODM (Object Data Modeling) library for MongoDB and Node.js
// It makes interacting with our database much easier by letting us define schemas
import mongoose from "mongoose";

// Import our custom route handlers
import authRoutes from "../routes/auth.js";
import checkRoutes from "../routes/check.js";
import historyRoutes from "../routes/history.js";

// Initialize the Express application
const app = express();

// Set the port the server will run on.
// process.env.PORT is usually provided by the hosting provider (like Render, Heroku).
// If not provided, we default to 3001.
const PORT = process.env.PORT || 3001;

// ---------------------------------------------------------------------------
// MIDDLEWARE CONFIGURATION
// Middleware are functions that run BEFORE our route handlers.
// They process the incoming requests (e.g., parsing data, checking headers).
// ---------------------------------------------------------------------------

// 1. Setup CORS
// We need this because our React app runs on port 5173 locally, but this server runs on 3001.
// Browsers block requests between different ports/domains by default for security.
app.use(cors({
  origin: process.env.CLIENT_URL || "http://localhost:5173",
  credentials: true, // Allows sending cookies/authorization headers
}));

// 2. Parse JSON bodies
// This tells Express to look at requests where the Content-Type is 'application/json'
// and parse the body into a JavaScript object (available at req.body).
// We limit the payload size to 1mb to prevent abuse (e.g., someone sending massive payloads).
app.use(express.json({ limit: "1mb" }));

// ---------------------------------------------------------------------------
// ROUTE REGISTRATION
// Routes map a specific URL path (e.g., '/api/auth') to a set of logic.
// ---------------------------------------------------------------------------

// Any request starting with '/api/auth' will be handled by the logic inside authRoutes
app.use("/api/auth", authRoutes);

// Any request starting with '/api/check' will be handled by the logic inside checkRoutes
app.use("/api/check", checkRoutes);

// Any request starting with '/api/history' will be handled by the logic inside historyRoutes
app.use("/api/history", historyRoutes);

/*
PURPOSE:
Health check endpoint.

INPUT:
None

OUTPUT:
A simple JSON object indicating the server is running.

WHY THIS EXISTS:
Used by hosting providers (and developers) to verify the server is up and didn't crash.
*/
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", service: "express-proxy" });
});

// ---------------------------------------------------------------------------
// DATABASE CONNECTION & SERVER START
// ---------------------------------------------------------------------------

// Get the MongoDB connection string.
// If not provided in .env, default to a local MongoDB instance.
const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017/newschecker";

/*
PURPOSE:
Connect to the database and start the server.

WHY THIS EXISTS:
We want to establish a database connection before we start accepting requests.
If the database connection fails, we log the error but start the server anyway so the health check endpoint still works.
*/
mongoose
  .connect(MONGODB_URI)
  .then(() => {
    // Step 1: Database connection successful
    console.log("Connected to MongoDB.");
    
    // Step 2: Start the web server
    app.listen(PORT, () => {
      console.log(`Express server running on http://localhost:${PORT}`);
    });
  })
  .catch((error) => {
    // Step 1: Database connection failed
    console.error("MongoDB connection failed:", error.message);
    
    // Step 2: Start anyway so health checks work during development without MongoDB
    app.listen(PORT, () => {
      console.log(`Express server running on http://localhost:${PORT} (no database)`);
    });
  });

// Export the Express app
// This is specifically required for serverless deployments like Vercel,
// which need to import the app directly rather than having it listen on a port.
export default app;
