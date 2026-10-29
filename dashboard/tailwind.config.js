/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        hazard: {
          low: "#22c55e",
          moderate: "#eab308",
          high: "#f97316",
          critical: "#ef4444",
          extreme: "#7f1d1d"
        }
      }
    },
  },
  plugins: [],
};
