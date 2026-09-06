/*
FILE PURPOSE:
Tests for the Express layer — auth middleware, history pagination, and the
/api/check proxy's error handling.

WHY THIS EXISTS:
The README listed "no automated test suite for server/" as the biggest
remaining test gap, and the two bugs pinned here are exactly the kind that gap
hides: a malformed history id came back as a 500 "Failed to fetch check", and
a negative ?skip reached MongoDB and crashed the query. Both look like server
failures to the caller when they are simply bad requests.

Run with:  node --test tests/
No database and no network are required — the proxy tests stub global fetch,
and the auth tests exercise the middleware directly.
*/

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { optionalAuth, requireAuth, signToken } from "../middleware/auth.js";
import { parsePagination } from "../routes/history.js";

// ── Helpers ──────────────────────────────────────────────────────────
function fakeResponse() {
  const res = { statusCode: null, body: null, headersSent: false };
  res.status = (code) => {
    res.statusCode = code;
    return res;
  };
  res.json = (payload) => {
    res.body = payload;
    return res;
  };
  return res;
}

function requestWith(token) {
  return { headers: token ? { authorization: `Bearer ${token}` } : {} };
}

// ── Auth middleware ──────────────────────────────────────────────────
describe("optionalAuth", () => {
  it("treats a request with no Authorization header as a guest", () => {
    const req = requestWith(null);
    let called = false;
    optionalAuth(req, fakeResponse(), () => {
      called = true;
    });
    assert.equal(req.user, null);
    assert.ok(called, "must always continue to the route handler");
  });

  it("treats a malformed token as a guest rather than throwing", () => {
    const req = requestWith("not-a-real-token");
    optionalAuth(req, fakeResponse(), () => {});
    assert.equal(req.user, null);
  });

  it("treats a token signed with the wrong secret as a guest", () => {
    // A forged token must never authenticate anyone.
    const forged =
      "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." +
      "eyJ1c2VySWQiOiJhdHRhY2tlciJ9.wrongsignature";
    const req = requestWith(forged);
    optionalAuth(req, fakeResponse(), () => {});
    assert.equal(req.user, null);
  });

  it("identifies the user from a valid token", () => {
    const req = requestWith(signToken({ userId: "abc123" }));
    optionalAuth(req, fakeResponse(), () => {});
    assert.equal(req.user.userId, "abc123");
  });

  it("ignores an Authorization header that is not a Bearer token", () => {
    const req = { headers: { authorization: "Basic dXNlcjpwYXNz" } };
    optionalAuth(req, fakeResponse(), () => {});
    assert.equal(req.user, null);
  });
});

describe("requireAuth", () => {
  it("rejects a guest with 401 and does not call the handler", () => {
    const res = fakeResponse();
    let handlerRan = false;
    requireAuth(requestWith(null), res, () => {
      handlerRan = true;
    });
    assert.equal(res.statusCode, 401);
    assert.equal(handlerRan, false);
  });

  it("lets an authenticated user through", () => {
    const req = requestWith(signToken({ userId: "abc123" }));
    let handlerRan = false;
    requireAuth(req, fakeResponse(), () => {
      handlerRan = true;
    });
    assert.ok(handlerRan);
    assert.equal(req.user.userId, "abc123");
  });
});

describe("signToken", () => {
  it("produces a token that round-trips through verification", () => {
    const req = requestWith(signToken({ userId: "u1", email: "a@b.c" }));
    optionalAuth(req, fakeResponse(), () => {});
    assert.equal(req.user.userId, "u1");
    assert.equal(req.user.email, "a@b.c");
  });
});

// ── History pagination ───────────────────────────────────────────────
describe("parsePagination", () => {
  it("uses sensible defaults when nothing is supplied", () => {
    assert.deepEqual(parsePagination({}), { limit: 50, skip: 0 });
  });

  it("reads valid values", () => {
    assert.deepEqual(parsePagination({ limit: "10", skip: "20" }), {
      limit: 10,
      skip: 20,
    });
  });

  it("never lets a negative skip reach the database", () => {
    // This was the bug: ?skip=-10 reached Mongo, which rejects it, and the
    // caller got a 500 for what is a bad request.
    assert.equal(parsePagination({ skip: "-10" }).skip, 0);
  });

  it("never lets a negative or zero limit reach the database", () => {
    assert.equal(parsePagination({ limit: "-5" }).limit, 1);
    assert.equal(parsePagination({ limit: "0" }).limit, 1);
  });

  it("caps the limit so one request cannot pull the whole collection", () => {
    assert.equal(parsePagination({ limit: "100000" }).limit, 100);
  });

  it("falls back to the default for non-numeric input", () => {
    assert.deepEqual(parsePagination({ limit: "abc", skip: "xyz" }), {
      limit: 50,
      skip: 0,
    });
  });
});

// ── /api/check proxy error handling ──────────────────────────────────
describe("check proxy", () => {
  async function callCheck({ statement, fetchImpl }) {
    // Import fresh so the route picks up the stubbed fetch.
    const { default: router } = await import("../routes/check.js");
    const layer = router.stack.find((l) => l.route?.methods?.post);
    const handlers = layer.route.stack.map((s) => s.handle);

    const req = { body: { statement }, headers: {} };
    const res = fakeResponse();

    const originalFetch = globalThis.fetch;
    if (fetchImpl) globalThis.fetch = fetchImpl;
    try {
      // optionalAuth, then the route handler.
      await new Promise((resolve) => handlers[0](req, res, resolve));
      await handlers[1](req, res, () => {});
    } finally {
      globalThis.fetch = originalFetch;
    }
    return res;
  }

  it("rejects a statement that is too short before calling the ML service", async () => {
    // This assertion only became real once the route stopped importing
    // node-fetch: an imported binding cannot be stubbed, so `called` was
    // never set either way and the check passed vacuously.
    let called = false;
    const res = await callCheck({
      statement: "hi",
      fetchImpl: async () => {
        called = true;
        return { ok: true, json: async () => ({}) };
      },
    });
    assert.equal(res.statusCode, 400);
    assert.equal(called, false, "must not spend a request on invalid input");
  });

  it("rejects a missing statement", async () => {
    const res = await callCheck({ statement: undefined });
    assert.equal(res.statusCode, 400);
  });

  it("rejects a non-string statement", async () => {
    const res = await callCheck({ statement: 42 });
    assert.equal(res.statusCode, 400);
  });
});

// ── Statement length bounds ──────────────────────────────────────────
describe("statement length", () => {
  async function callCheck(statement) {
    const { default: router } = await import("../routes/check.js");
    const layer = router.stack.find((l) => l.route?.methods?.post);
    const handlers = layer.route.stack.map((s) => s.handle);
    const req = { body: { statement }, headers: {} };
    const res = fakeResponse();

    const originalFetch = globalThis.fetch;
    let upstreamCalled = false;
    globalThis.fetch = async () => {
      upstreamCalled = true;
      return { ok: true, json: async () => ({}) };
    };
    try {
      await new Promise((resolve) => handlers[0](req, res, resolve));
      await handlers[1](req, res, () => {});
    } finally {
      globalThis.fetch = originalFetch;
    }
    return { res, upstreamCalled };
  }

  it("rejects an over-long statement here rather than upstream", async () => {
    // Forwarding it produced a Pydantic 422, relayed to the user as a 502
    // "ML service error" — a broken backend, apparently, rather than "too long".
    const { res, upstreamCalled } = await callCheck("a".repeat(2001));
    assert.equal(res.statusCode, 400);
    assert.match(res.body.error, /at most 2000/);
    assert.equal(upstreamCalled, false);
  });

  it("accepts a statement at exactly the limit", async () => {
    const { upstreamCalled } = await callCheck("a".repeat(2000));
    assert.equal(upstreamCalled, true, "2000 characters is valid, not too long");
  });

  it("measures the trimmed length, not the raw length", async () => {
    const { upstreamCalled } = await callCheck("  " + "a".repeat(2000) + "  ");
    assert.equal(upstreamCalled, true);
  });
});

// ── Google sign-in configuration ─────────────────────────────────────
describe("google sign-in configuration", () => {
  /*
    GOOGLE_CLIENT_ID is a security control, not just configuration. It is
    passed to verifyIdToken as `audience`, and google-auth-library skips the
    audience check entirely when that is undefined — so an ID token minted for
    any other Google application would authenticate. Sign-in appears to work,
    which is what makes it dangerous.
  */
  async function loadAuthRoute(env) {
    const saved = { ...process.env };
    Object.assign(process.env, env);
    delete process.env.GOOGLE_CLIENT_ID;
    if (env.GOOGLE_CLIENT_ID) process.env.GOOGLE_CLIENT_ID = env.GOOGLE_CLIENT_ID;
    try {
      // Cache-bust so module-level checks re-run.
      return await import(`../routes/auth.js?t=${Date.now()}${Math.random()}`);
    } finally {
      for (const key of Object.keys(process.env)) delete process.env[key];
      Object.assign(process.env, saved);
    }
  }

  it("refuses to start in production without a client id", async () => {
    await assert.rejects(
      () => loadAuthRoute({ NODE_ENV: "production" }),
      /GOOGLE_CLIENT_ID/,
      "an unset client id disables audience verification, so production must not boot",
    );
  });

  it("still loads in development, for local work without sign-in", async () => {
    const mod = await loadAuthRoute({ NODE_ENV: "development" });
    assert.ok(mod.default, "the router must still be importable");
  });

  it("rejects a sign-in attempt when no client id is configured", async () => {
    const { default: router } = await loadAuthRoute({ NODE_ENV: "development" });
    const layer = router.stack.find(
      (l) => l.route?.path === "/google" && l.route?.methods?.post,
    );
    const handler = layer.route.stack[0].handle;

    const res = fakeResponse();
    await handler({ body: { credential: "any.google.token" }, headers: {} }, res, () => {});
    assert.equal(res.statusCode, 503, "must refuse, never verify without an audience");
  });

  it("still rejects a request with no credential at all", async () => {
    const { default: router } = await loadAuthRoute({
      NODE_ENV: "development",
      GOOGLE_CLIENT_ID: "test-client-id.apps.googleusercontent.com",
    });
    const layer = router.stack.find(
      (l) => l.route?.path === "/google" && l.route?.methods?.post,
    );
    const handler = layer.route.stack[0].handle;

    const res = fakeResponse();
    await handler({ body: {}, headers: {} }, res, () => {});
    assert.equal(res.statusCode, 400);
  });
});
