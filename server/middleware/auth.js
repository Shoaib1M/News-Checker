import jwt from "jsonwebtoken";

const JWT_SECRET = process.env.JWT_SECRET || "newschecker-jwt-secret-change-me";

/**
 * JWT authentication middleware.
 *
 * - If a valid token is present → sets req.user = { userId, email, name }
 * - If no token is present → sets req.user = null (guest)
 *
 * Use `requireAuth` when the route must have a logged-in user.
 */
export function optionalAuth(req, _res, next) {
  const header = req.headers.authorization;

  if (!header || !header.startsWith("Bearer ")) {
    req.user = null;
    return next();
  }

  try {
    const token = header.split(" ")[1];
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
  } catch {
    req.user = null;
  }

  next();
}

export function requireAuth(req, res, next) {
  optionalAuth(req, res, () => {
    if (!req.user) {
      return res.status(401).json({ error: "Authentication required." });
    }
    next();
  });
}

export function signToken(payload) {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: "7d" });
}
