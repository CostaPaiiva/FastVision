import * as React from "react";

type Variant = "default" | "outline";

export function Button({
  className = "",
  variant = "default",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  const base =
    "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors disabled:opacity-60 disabled:pointer-events-none px-4 py-2";
  const styles =
    variant === "outline"
      ? "border border-zinc-200 bg-white hover:bg-zinc-50 text-zinc-900"
      : "bg-zinc-900 hover:bg-zinc-800 text-white";

  return <button className={`${base} ${styles} ${className}`} {...props} />;
}
