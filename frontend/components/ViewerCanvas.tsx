"use client";

import { useEffect, useMemo, useRef } from "react";

export type Detection = {
  cls: number;
  name: string;
  conf: number;
  box: number[]; // [x1,y1,x2,y2]
};

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n));
}

export default function ViewerCanvas({
  imageUrl,
  detections,
  selectedIndex
}: {
  imageUrl: string | null;
  detections: Detection[];
  selectedIndex: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const selected = useMemo(() => {
    if (selectedIndex === null) return null;
    return detections[selectedIndex] ?? null;
  }, [selectedIndex, detections]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!imageUrl) {
      // Placeholder
      canvas.width = 900;
      canvas.height = 500;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = "14px ui-sans-serif";
      ctx.fillStyle = "#666";
      ctx.fillText("Nenhuma imagem carregada ainda.", 20, 40);
      return;
    }

    const img = new Image();
    img.onload = () => {
      // Ajusta canvas ao tamanho da imagem (mantém qualidade)
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Draw boxes
      for (let i = 0; i < detections.length; i++) {
        const d = detections[i];
        const [x1, y1, x2, y2] = d.box;
        const isSel = selectedIndex === i;

        ctx.lineWidth = isSel ? 4 : 2;
        ctx.strokeStyle = isSel ? "#111" : "#00aa00";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = `${d.name} ${d.conf.toFixed(2)}`;
        ctx.font = "14px ui-sans-serif";
        const metrics = ctx.measureText(label);
        const tw = metrics.width;
        const th = 18;

        const bx = clamp(x1, 0, canvas.width - tw - 10);
        const by = clamp(y1 - th - 6, 0, canvas.height - th);

        // label background
        ctx.fillStyle = isSel ? "rgba(17,17,17,0.85)" : "rgba(0,170,0,0.75)";
        ctx.fillRect(bx, by, tw + 10, th + 6);

        // label text
        ctx.fillStyle = "#fff";
        ctx.fillText(label, bx + 5, by + 18);
      }

      // Selected details (optional)
      if (selected) {
        // Could add a subtle mark
      }
    };
    img.src = imageUrl;
  }, [imageUrl, detections, selectedIndex, selected]);

  return (
    <div className="w-full overflow-auto border border-zinc-200 rounded-md bg-white">
      <canvas ref={canvasRef} className="block max-w-none" />
    </div>
  );
}
