import type { NextConfig } from "next";
const config: NextConfig = {
  distDir: process.env.SUPERTRADER_DIST_DIR ?? ".next-supertrader",
  poweredByHeader: false,
};
export default config;
