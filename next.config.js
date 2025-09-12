/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  env: {
    BACKEND_API_URL: process.env.BACKEND_API_URL || 'http://backend:8080',
    RETRIEVER_URL: process.env.RETRIEVER_URL || 'http://backend:8080',
  },
};

module.exports = nextConfig;