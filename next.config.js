/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  env: {
    BACKEND_API_URL: process.env.BACKEND_API_URL || 'http://api:8080',
    RETRIEVER_URL: process.env.RETRIEVER_URL || 'http://retriever:8081',
  },
};

module.exports = nextConfig;