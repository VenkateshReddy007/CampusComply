import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": resolve(root) } },
  test: { environment: "jsdom", setupFiles: ["./vitest.setup.ts"], include: ["**/*.test.ts", "**/*.test.tsx"] },
});
