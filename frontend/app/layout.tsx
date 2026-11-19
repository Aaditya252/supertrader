import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "Supertrader · SOL research terminal", description: "Auditable Solana signals, forward validation and paper trading." };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
