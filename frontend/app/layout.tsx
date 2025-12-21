import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Avatar Renderer MCP — Futuristic Demo',
  description: 'A futuristic, deployable Vercel demo: choose an avatar, add voice/script, generate, and embed in your React/Vercel app.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
