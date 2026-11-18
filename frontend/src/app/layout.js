import "../styles/globals.css";
import { Fraunces, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import SmoothScroll from "../components/SmoothScroll";

const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-display", style: ["normal", "italic"] });
const spaceGrotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-sans" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata = {
  title: "Pulse AI",
  description: "Personalized, explainable news & market recommendations",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${fraunces.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable}`}>
      <body className="bg-slate-50 font-sans text-slate-900">
        <SmoothScroll>{children}</SmoothScroll>
      </body>
    </html>
  );
}