import * as React from "react";
import { cn } from "@/lib/cn";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea ref={ref} className={cn("min-h-24 w-full resize-y rounded-md border border-line bg-white px-3 py-2 text-sm text-ink placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-700", className)} {...props} />
  ),
);
Textarea.displayName = "Textarea";
