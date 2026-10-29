import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multi-Hazard EEWS | AI-Driven Flood & Drought Early Warning",
  description: "Production-ready, leakage-safe geospatial early warning system for hydroclimatic hazards.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 min-h-screen antialiased selection:bg-cyan-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
