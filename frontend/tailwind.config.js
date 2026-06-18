/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['"DM Sans"', 'sans-serif'],
      },
      colors: {
        ink:    '#0e0e0e',
        paper:  '#f5f2eb',
        ash:    '#c8c4bc',
        dim:    '#6b6760',
        accent: '#d4622a',
        soft:   '#e8e4dc',
      },
      keyframes: {
        blink: { '0%,100%': { opacity: '1' }, '50%': { opacity: '0' } },
        fadein: { from: { opacity: '0', transform: 'translateY(6px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
      },
      animation: {
        blink:  'blink 1s step-end infinite',
        fadein: 'fadein 0.3s ease forwards',
      },
    },
  },
  plugins: [],
}
