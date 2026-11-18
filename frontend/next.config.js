/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // The browser calls this app's own /api/* (same-origin — no CORS, no
  // "localhost doesn't mean what I think it means" issues in cloud/sandboxed
  // dev environments). Next's own server then forwards it to the real
  // backend, which it CAN reach directly since that's a server-to-server
  // call, not a browser one.
  // BACKEND_INTERNAL_URL: "http://localhost:4000" for local/native dev,
  // "http://backend:4000" when running via docker-compose (service name).
  async rewrites() {
    const backendUrl = process.env.BACKEND_INTERNAL_URL || "http://localhost:4000";
    return [{ source: "/api/:path*", destination: `${backendUrl}/api/:path*` }];
  },
};

module.exports = nextConfig;
