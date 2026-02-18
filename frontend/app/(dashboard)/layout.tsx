"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { clearTokens, isLoggedIn } from "@/lib/api";

function NavLink({ href, label }: { href: string; label: string }) {
  const pathname = usePathname();
  const active = pathname === href;
  return (
    <Link
      href={href}
      className={[
        "text-sm px-3 py-2 rounded-md",
        active ? "bg-zinc-900 text-white" : "text-zinc-700 hover:text-zinc-900 hover:bg-zinc-100",
      ].join(" ")}
    >
      {label}
    </Link>
  );
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();

  function logout() {
    clearTokens();
    router.push("/login");
  }

  // Guarda simples no client
  if (typeof window !== "undefined" && !isLoggedIn()) {
    router.push("/login");
    return null;
  }

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white">
        <div className="max-w-5xl mx-auto p-4 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold">SkillUp YOLO Local</span>
            <span className="text-xs text-zinc-500">Windows • Local</span>
          </div>

          <nav className="flex items-center gap-1">
            <NavLink href="/infer" label="Inferir" />
            <NavLink href="/history" label="Histórico" />
            <NavLink href="/about" label="About" />
          </nav>

          <Button variant="outline" onClick={logout}>
            Sair
          </Button>
        </div>
      </header>

      <main className="max-w-5xl mx-auto p-4">{children}</main>
    </div>
  );
}
