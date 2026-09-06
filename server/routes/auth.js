/*
FILE PURPOSE:
This file handles authentication routes, specifically logging in with Google.

FLOW:
1. Receives a Google ID token from the frontend.
2. Verifies that token directly with Google's servers to ensure it's authentic.
3. Checks if the user already exists in our MongoDB database.
4. If they do, updates their profile. If not, creates a new user.
5. Generates a custom JWT (JSON Web Token) for the user to use in our app.
6. Sends the token and user profile back to the frontend.

USED BY:
- The React frontend when a user clicks the "Sign in with Google" button.
*/

import { Router } from "express";
import { OAuth2Client } from "google-auth-library";
import User from "../models/User.js";
import { signToken, requireAuth } from "../middleware/auth.js";

// Create a new Express Router to organize our routes
const router = Router();

// Our Google Client ID (from the Google Cloud Console).
//
// This is a security control, not just configuration. It is passed to
// verifyIdToken as `audience`, and google-auth-library SKIPS the audience
// check entirely when that value is undefined:
//
//     if (typeof requiredAudience !== 'undefined' && requiredAudience !== null)
//         ...check aud...                        (oauth2client.js)
//
// So with GOOGLE_CLIENT_ID unset, a valid Google ID token issued for ANY
// other application would be accepted, and whoever holds it signs in as that
// user. Sign-in appears to work, which is what makes it dangerous.
//
// Fail fast in production, the same treatment JWT_SECRET gets in
// middleware/auth.js; warn in development, where sign-in is often simply not
// configured.
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;

if (!GOOGLE_CLIENT_ID) {
  const message =
    "GOOGLE_CLIENT_ID is not set. Google sign-in cannot verify which app a " +
    "token was issued for, so tokens from any Google application would be " +
    "accepted.";
  if (process.env.NODE_ENV === "production") {
    throw new Error(message);
  }
  console.warn(`  ⚠  ${message} Sign-in is disabled until it is set.`);
}

// Initialize the Google OAuth client
const googleClient = new OAuth2Client(GOOGLE_CLIENT_ID);

/*
PURPOSE:
Handle the POST request when a user tries to log in with Google.

INPUT:
req.body.credential (The Google ID token string)

OUTPUT:
JSON response containing { token, user } on success, or an error message on failure.

WHY THIS EXISTS:
We don't manage passwords ourselves. We let Google authenticate the user, 
and then we create a session (via JWT) in our own app.
*/
router.post("/google", async (req, res) => {
  try {
    const { credential } = req.body;

    // Step 1: Validate input
    if (!credential) {
      return res.status(400).json({ error: "Missing Google credential token." });
    }

    // Refuse rather than verify without an audience. Reaching verifyIdToken
    // with audience undefined would accept a token minted for any other
    // Google app.
    if (!GOOGLE_CLIENT_ID) {
      return res.status(503).json({
        error: "Google sign-in is not configured on this server.",
      });
    }

    // Step 2: Verify the token with Google
    // This makes a network request to Google to ensure the token wasn't forged.
    const ticket = await googleClient.verifyIdToken({
      idToken: credential,
      audience: GOOGLE_CLIENT_ID, // Ensure the token was meant for OUR app
    });

    // Step 3: Extract user data from the verified token
    const payload = ticket.getPayload();
    const { sub: googleId, email, name, picture } = payload; // 'sub' is Google's unique ID for the user

    // Step 4: Find or create the user in our MongoDB database
    let user = await User.findOne({ googleId });

    if (!user) {
      // If the user doesn't exist, create a new record
      user = await User.create({
        googleId,
        email,
        name,
        avatar: picture || "", // Fallback to empty string if no picture
      });
    } else {
      // If they do exist, update their name and avatar in case they changed it on Google
      user.name = name;
      user.avatar = picture || "";
      await user.save();
    }

    // Step 5: Generate our own JWT
    // We don't want to pass Google's token around forever; we use our own.
    const token = signToken({
      userId: user._id.toString(), // Our internal MongoDB ID
      email: user.email,
      name: user.name,
    });

    // Step 6: Send the successful response
    res.json({
      token, // The frontend will save this token (e.g., in localStorage)
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        avatar: user.avatar,
      },
    });
  } catch (error) {
    // If anything goes wrong (like an expired Google token), catch the error here
    console.error("Google auth error:", error.message);
    res.status(401).json({ error: "Invalid Google credential." });
  }
});

/*
PURPOSE:
Fetch the currently logged-in user's profile data.

INPUT:
Requires a valid JWT token in the Authorization header (handled by `requireAuth`).

OUTPUT:
JSON response with the user's profile data.

WHY THIS EXISTS:
When a user refreshes the page, the React app only has the JWT token. 
It calls this endpoint to fetch the user's actual name, email, and avatar again.
*/
router.get("/me", requireAuth, async (req, res) => {
  try {
    // Look up the user by the ID stored in the token (req.user is set by requireAuth)
    // .select("-__v") tells Mongoose NOT to return the internal version field
    const user = await User.findById(req.user.userId).select("-__v");

    if (!user) {
      return res.status(404).json({ error: "User not found." });
    }

    // Send back the user profile
    res.json({
      id: user._id,
      name: user.name,
      email: user.email,
      avatar: user.avatar,
    });
  } catch (error) {
    console.error("Get user error:", error.message);
    res.status(500).json({ error: "Failed to fetch user." });
  }
});

export default router;
