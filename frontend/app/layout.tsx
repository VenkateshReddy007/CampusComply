import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CampusReg | CampusComply",
  description: "Find campus regulations with page-level evidence from your indexed documents.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
