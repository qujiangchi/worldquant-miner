import type { Metadata } from 'next'
import './globals.css'
import { ThemeProvider } from '@/components/theme-provider'
import { FloatingDock } from '@/components/ui/floating-dock'
import { sharedNavItems } from '@/components/ui/shared-navigation'
import { Analytics } from "@vercel/analytics/react"

export const metadata: Metadata = {
  title: 'WorldQuant Brain',
  description: 'WorldQuant Brain Dashboard',
  generator: 'Next.js',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning={true}>
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <FloatingDock items={sharedNavItems} />
        </ThemeProvider>
        <Analytics mode="production" />
      </body>
    </html>
  )
}
