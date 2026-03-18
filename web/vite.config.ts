import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

const devApiTarget = process.env.VITE_DEV_API_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  base: "/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: devApiTarget,
        changeOrigin: true,
      },
      "/health": {
        target: devApiTarget,
        changeOrigin: true,
      },
    },
  },
});
