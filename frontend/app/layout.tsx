import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SkillUp YOLO Local",
  description: "Detecção de imagens local com YOLO + FastAPI + Next.js",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
