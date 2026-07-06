# MVP image used by docker-compose: serves the Vite dev server on 5173
# (matches the compose `5173:5173` mapping and the "React SPA — Vite dev server" contract).
# For a static production deployment, see the nginx multi-stage variant in BUILD_GUIDE §50.
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . .
EXPOSE 5173
# `npm run dev` == `vite --host --port 5173`
CMD ["npm", "run", "dev"]
