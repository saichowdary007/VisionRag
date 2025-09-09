const guessApiBase = () => {
  const envBase = (import.meta as any).env?.VITE_API_BASE;
  if (envBase) return envBase;
  if (typeof window !== "undefined") {
    const url = new URL(window.location.href);
    if (url.port === "5173") {
      return url.origin.replace(":5173", ":8080");
    }
  }
  return "http://localhost:8080";
};

export const API_BASE = guessApiBase();

export async function ingestFromUrl(pdf_url: string) {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pdf_url }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function ingestFile(file: File) {
  const form = new FormData();
  form.append("pdf_file", file);
  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function askStream(
  question: string,
  opts: { k?: number; m?: number; doc_id?: string } = {}
): Promise<{ text: string; images: string[] }>{
  // Kept for backwards compatibility. Uses the generator under the hood.
  let text = "";
  let images: string[] = [];
  for await (const evt of askStreamGen(question, opts)) {
    if (evt.type === "meta") images = evt.images;
    if (evt.type === "delta") text += evt.text;
  }
  return { text, images };
}

export type SourceItem = {
  doc_id?: string;
  page?: number;
  score?: number;
  image_url: string;
  heatmap_url?: string | null;
};

export type AskStreamEvent =
  | { type: "meta"; images: string[]; sources?: SourceItem[] }
  | { type: "delta"; text: string };

export async function* askStreamGen(
  question: string,
  opts: { k?: number; m?: number; doc_id?: string } = {}
): AsyncGenerator<AskStreamEvent, void, void> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, ...opts }),
  });
  if (!res.ok || !res.body) throw new Error(await res.text());
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let first = true;
  let buffered = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    if (first) {
      first = false;
      const nl = chunk.indexOf("\n");
      if (nl !== -1) {
        try {
          const meta = JSON.parse(chunk.slice(0, nl));
          if (
            (meta?.type === "images" || meta?.type === "meta") &&
            Array.isArray(meta.images)
          ) {
            const sources = Array.isArray(meta.sources) ? meta.sources : undefined;
            yield { type: "meta", images: meta.images, sources };
            const rest = chunk.slice(nl + 1);
            if (rest) {
              yield { type: "delta", text: rest };
            }
            continue;
          }
        } catch {}
      }
    }
    buffered += chunk;
    if (buffered) {
      yield { type: "delta", text: buffered };
      buffered = "";
    }
  }
}
