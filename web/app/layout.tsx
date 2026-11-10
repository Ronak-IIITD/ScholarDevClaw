import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ScholarDevClaw Dashboard",
  description: "Research-to-Code Integration Dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col">{children}</body>
    </html>
  );
}
