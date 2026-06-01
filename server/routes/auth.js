import { Router } from "express";
import { OAuth2Client } from "google-auth-library";
import User from "../models/User.js";
import { signToken, requireAuth } from "../middleware/auth.js";

const router = Router();
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const googleClient = new OAuth2Client(GOOGLE_CLIENT_ID);

/**
 * POST /api/auth/google
 * Body: { credential: "<Google ID token>" }
 *
 * Verifies the Google ID token, creates or finds the user in MongoDB,
 * and returns a signed JWT + user profile.
 */
router.post("/google", async (req, res) => {
  try {
    const { credential } = req.body;

    if (!credential) {
      return res.status(400).json({ error: "Missing Google credential token." });
    }

    // Verify the token with Google
    const ticket = await googleClient.verifyIdToken({
      idToken: credential,
      audience: GOOGLE_CLIENT_ID,
    });

    const payload = ticket.getPayload();
    const { sub: googleId, email, name, picture } = payload;

    // Find or create user in MongoDB
    let user = await User.findOne({ googleId });

    if (!user) {
      user = await User.create({
        googleId,
        email,
        name,
        avatar: picture || "",
      });
    } else {
      // Update profile in case name/avatar changed
      user.name = name;
      user.avatar = picture || "";
      await user.save();
    }

    // Sign JWT
    const token = signToken({
      userId: user._id.toString(),
      email: user.email,
      name: user.name,
    });

    res.json({
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        avatar: user.avatar,
      },
    });
  } catch (error) {
    console.error("Google auth error:", error.message);
    res.status(401).json({ error: "Invalid Google credential." });
  }
});

/**
 * GET /api/auth/me
 * Returns the current authenticated user's profile.
 */
router.get("/me", requireAuth, async (req, res) => {
  try {
    const user = await User.findById(req.user.userId).select("-__v");

    if (!user) {
      return res.status(404).json({ error: "User not found." });
    }

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
