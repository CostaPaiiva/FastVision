"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { apiPost, setTokens } from "@/lib/api";

type Mode = "login" | "register";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const title = useMemo(() => (mode === "login" ? "Entrar" : "Criar conta"), [mode]);

  async function submit() {
    setError(null);
    setLoading(true);
    try {
      const path = mode === "login" ? "/auth/login" : "/auth/register";
      const data = await apiPost(path, { email, password }, { auth: false });
      setTokens(data.access_token, data.refresh_token);
      router.push("/infer");
    } catch (e: any) {
      setError(e?.message ?? "Falha");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold">{title}</h1>
            <button
              className="text-sm text-zinc-600 hover:text-zinc-900"
              onClick={() => setMode(mode === "login" ? "register" : "login")}
            >
              {mode === "login" ? "Registrar" : "Já tenho conta"}
            </button>
          </div>
          <p className="text-sm text-zinc-600">
            Tokens ficam no LocalStorage (modo local). Para produção, use cookies httpOnly.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Email</Label>
            <Input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="voce@exemplo.com" />
          </div>
          <div className="space-y-2">
            <Label>Senha</Label>
            <Input
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="mínimo 6 caracteres"
              type="password"
            />
          </div>

          {error && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 p-3 rounded-md">
              {error}
            </div>
          )}

          <Button className="w-full" onClick={submit} disabled={loading}>
            {loading ? "Enviando..." : title}
          </Button>

          <div className="text-xs text-zinc-500">
            Dica: Se der 401 em rotas privadas, faça logout e logue novamente.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
