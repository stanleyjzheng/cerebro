import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config) => {
    config.infrastructureLogging = { level: "error" }; // Suppresses warnings
    return config;
  },
};

export default nextConfig;
