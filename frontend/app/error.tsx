"use client";

import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function ErrorBoundary({ retry }: { error: unknown; retry: () => void }) {
  return (
    <main className="route-error" role="alert">
      <div className="empty-icon"><AlertCircle size={20} /></div>
      <h1>This workspace could not load</h1>
      <p>Reload the view and retry. The error has not been shown as a policy answer.</p>
      <Button onClick={retry}><RefreshCw size={14} />Retry</Button>
    </main>
  );
}
