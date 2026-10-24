import * as React from "react";
import { cn } from "@/lib/cn";

export function Badge({ className, ...props }: React.HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn("inline-flex items-center rounded-sm border border-line px-2 py-0.5 text-xs font-medium", className)} {...props} />;
}
