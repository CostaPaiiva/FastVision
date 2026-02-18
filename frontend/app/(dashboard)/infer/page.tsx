"use client";

import { useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import ViewerCanvas, { Detection } from "@/components/ViewerCanvas";
import { apiGet, apiUpload } from "@/lib/api";

type InferResponse = {
  id: string;
  model_name: string;
  imgsz: number;
  conf: number;
  iou: number;
  detections: Detection[];
};

export default function InferPage() {
  const fileRef = useRef<HTMLInputElement | null>(null);

  const [imgsz, setImgsz] = useState<number>(640);
  const [conf, setConf] = useState<number>(0.25);
  const [iou, setIou] = useState<number>(0.7);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [resultId, setResultId] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const selected = useMemo(() => {
    if (selectedIdx === null) return null;
    return detections[selectedIdx] ?? null;
  }, [selectedIdx, detections]);

  async function infer() {
    setError(null);
    setSelectedIdx(null);

    const f = fileRef.current?.files?.[0];
    if (!f) {
      setError("Selecione uma imagem primeiro.");
      return;
    }

    setLoading(true);
    try {
      const data: InferResponse = await apiUpload("/infer/image", {
        file: f,
        imgsz: String(imgsz),
        conf: String(conf),
        iou: String(iou),
      });

      setResultId(data.id);
      setDetections(data.detections);

      // Para renderizar a imagem original no canvas, usamos URL local do browser.
      const localUrl = URL.createObjectURL(f);
      setImageUrl(localUrl);
    } catch (e: any) {
      setError(e?.message ?? "Falha na inferência.");
    } finally {
      setLoading(false);
    }
  }

  async function openAnnotated() {
    if (!resultId) return;
    // Abre a imagem anotada em nova aba (autorizada)
    // Criamos um link com token via fetch e blob
    try {
      const res = await apiGet(`/infer/result/${resultId}/image`, { raw: true });
      const blob = await (res as Response).blob();
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
    } catch (e: any) {
      setError(e?.message ?? "Falha ao abrir imagem anotada.");
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <h1 className="text-xl font-semibold">Inferência</h1>
          <p className="text-sm text-zinc-600">
            Upload de imagem (JPEG/PNG/WEBP até 10MB) e parâmetros leves por padrão.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <div className="md:col-span-2">
              <Label>Imagem</Label>
              <Input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" />
            </div>

            <div>
              <Label>imgsz</Label>
              <Input
                type="number"
                value={imgsz}
                onChange={(e) => setImgsz(Number(e.target.value))}
                min={320}
                max={1280}
                step={32}
              />
              <div className="text-xs text-zinc-500 mt-1">Recomendado: 640 (ou 512)</div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <Label>conf</Label>
                <Input
                  type="number"
                  value={conf}
                  onChange={(e) => setConf(Number(e.target.value))}
                  min={0.01}
                  max={1}
                  step={0.01}
                />
              </div>
              <div>
                <Label>iou</Label>
                <Input
                  type="number"
                  value={iou}
                  onChange={(e) => setIou(Number(e.target.value))}
                  min={0.01}
                  max={1}
                  step={0.01}
                />
              </div>
            </div>
          </div>

          {error && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 p-3 rounded-md">
              {error}
            </div>
          )}

          <div className="flex items-center gap-2">
            <Button onClick={infer} disabled={loading}>
              {loading ? "Inferindo..." : "Inferir"}
            </Button>
            <Button variant="outline" onClick={openAnnotated} disabled={!resultId}>
              Ver imagem anotada
            </Button>
            {resultId && (
              <div className="text-xs text-zinc-500">
                Result ID: <span className="font-mono">{resultId}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <Card className="lg:col-span-3">
          <CardHeader>
            <h2 className="text-base font-semibold">Viewer</h2>
            <p className="text-sm text-zinc-600">Canvas overlay (leve) desenhando boxes e labels.</p>
          </CardHeader>
          <CardContent>
            <ViewerCanvas imageUrl={imageUrl} detections={detections} selectedIndex={selectedIdx} />
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <h2 className="text-base font-semibold">Detecções</h2>
            <p className="text-sm text-zinc-600">Clique para destacar um box.</p>
          </CardHeader>
          <CardContent>
            {detections.length === 0 ? (
              <div className="text-sm text-zinc-500">Sem detecções ainda.</div>
            ) : (
              <div className="space-y-2">
                {detections.map((d, idx) => {
                  const active = idx === selectedIdx;
                  return (
                    <button
                      key={idx}
                      onClick={() => setSelectedIdx(active ? null : idx)}
                      className={[
                        "w-full text-left border rounded-md p-2 text-sm",
                        active ? "border-zinc-900 bg-zinc-50" : "border-zinc-200 hover:bg-white",
                      ].join(" ")}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium">{d.name}</span>
                        <span className="font-mono text-xs">{d.conf.toFixed(2)}</span>
                      </div>
                      <div className="text-xs text-zinc-500 font-mono">
                        box: [{d.box.map((v) => v.toFixed(0)).join(", ")}]
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {selected && (
              <div className="mt-3 text-xs text-zinc-600">
                <div className="font-semibold">Selecionado:</div>
                <div className="font-mono">
                  {selected.name} • conf {selected.conf.toFixed(2)}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
