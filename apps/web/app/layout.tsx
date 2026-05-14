import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Scientist Copilot",
  description: "AI-first scientific assistant foundation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
