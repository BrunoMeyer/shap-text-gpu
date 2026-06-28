import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
    base: "/bhmeyer/shap-text-gpu/",
    server: {
      proxy: {
        // Proxy frontend `/api` calls to the backend running on port 8000 in development
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          secure: false,
        },
      },
    },
})
