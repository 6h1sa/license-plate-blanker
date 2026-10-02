import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// 開発中（npm run dev）は API と画像を app.py（uv run app.py）へ中継する
const backend = "http://127.0.0.1:7900";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": backend, "/img": backend } },
  build: { outDir: "dist", emptyOutDir: true },
});
