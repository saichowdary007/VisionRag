/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // Avoid hardcoding server URLs at build time; rely on runtime env instead.
};

module.exports = nextConfig;
