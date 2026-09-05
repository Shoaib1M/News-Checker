import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server-only convenience: forwards /api/* requests to the local
// Node/Express server (server/api/index.js) so the client can call
// relative "/api/..." paths without CORS during `npm run dev`. This has
// no effect on the production build — deployed builds use VITE_API_URL
// (see App.jsx) or same-origin routing via vercel.json rewrites instead.
// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:3001",
        changeOrigin: true,
      },
    },
  },
});
