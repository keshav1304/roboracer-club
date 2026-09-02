import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RoboRacer Playground",
  description:
    "Interactive playground for AutoDRIVE RoboRacer — tune racing algorithms live and visualize sensor data.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
