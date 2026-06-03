/*
FILE PURPOSE:
This file contains custom Express middleware and helper functions for Authentication using JSON Web Tokens (JWT).

FLOW:
1. `signToken`: Creates a new JWT when a user logs in or signs up.
2. `optionalAuth`: Middleware that checks if a user sent a JWT. If they did, it identifies them. If not, it treats them as a guest.
3. `requireAuth`: Middleware that explicitly blocks the request if the user is not logged in.

USED BY:
- server/routes/auth.js (uses `signToken` when logging in a user)
- server/routes/check.js (uses `optionalAuth` so both guests and logged-in users can check statements)
- server/routes/history.js (uses `requireAuth` because guests don't have a history)
*/

// jsonwebtoken is a library that creates and verifies secure, signed tokens
import jwt from "jsonwebtoken";

// The secret key used to sign and verify our tokens.
// In production, this MUST be a long random string stored securely in environment variables.
const JWT_SECRET = process.env.JWT_SECRET || "newschecker-jwt-secret-change-me";

/*
PURPOSE:
Middleware that checks if a valid JWT is present in the request.
It allows the request to proceed regardless of whether the user is logged in or not.

INPUT:
req (request), res (response), next (function to pass control to the next middleware/route)

OUTPUT:
Sets `req.user` if a valid token is found. Otherwise, sets `req.user = null`.

WHY THIS EXISTS:
Sometimes we have routes (like /api/check) where guests can use the feature, but if a user IS logged in, we want to know who they are (e.g., to save the result to their history).
*/
export function optionalAuth(req, _res, next) {
  // Step 1: Look for the Authorization header in the incoming request
  // The standard format is: "Bearer <token>"
  const header = req.headers.authorization;

  // Step 2: If the header is missing or doesn't start with "Bearer "
  if (!header || !header.startsWith("Bearer ")) {
    req.user = null; // Treat them as a guest
    return next();   // Move to the next function
  }

  // Step 3: Try to extract and verify the token
  try {
    // Split "Bearer <token>" by the space and grab the second part
    const token = header.split(" ")[1];
    
    // verify() throws an error if the token is invalid or expired
    const decoded = jwt.verify(token, JWT_SECRET);
    
    // If successful, attach the decoded user data (like userId) to the request object
    req.user = decoded;
  } catch {
    // If verification fails (e.g., expired token), silently treat them as a guest
    req.user = null;
  }

  // Step 4: Move on to the route handler
  next();
}

/*
PURPOSE:
Strict middleware that demands the user be authenticated.

INPUT:
req, res, next

OUTPUT:
Passes to the next function if authorized. Sends a 401 Unauthorized error if not.

WHY THIS EXISTS:
Used to protect routes that absolutely require a user account, like viewing past history.
*/
export function requireAuth(req, res, next) {
  // We reuse our optionalAuth middleware to extract the token if it exists
  optionalAuth(req, res, () => {
    // If after running optionalAuth, req.user is still not set, they are not logged in
    if (!req.user) {
      return res.status(401).json({ error: "Authentication required." });
    }
    
    // If req.user exists, they are logged in. Allow the request to proceed.
    next();
  });
}

/*
PURPOSE:
Helper function to generate a new JWT.

INPUT:
payload: An object containing data we want to encode into the token (e.g., { userId: "123" })

OUTPUT:
A signed JWT string.

WHY THIS EXISTS:
When a user successfully logs in via Google, we generate this token and send it back to the React frontend. The frontend will then send this token with future requests to prove who they are.
*/
export function signToken(payload) {
  // Create a token that expires in 7 days
  return jwt.sign(payload, JWT_SECRET, { expiresIn: "7d" });
}
